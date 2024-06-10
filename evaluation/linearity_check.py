import sys
sys.path.append('.')

import numpy as np
import torch
from torch.cuda.amp import autocast
from datasets import load_dataset

from reward_sampling import RewardSampling
import data_loader
from tools import uncertainty

import argparse
import json
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--llm-dir', type=str, default='argsearch/llama-7b-sft-float32')
parser.add_argument('--rm-dir', type=str, default='argsearch/llama-7b-rm-float32')
parser.add_argument('--data-dir', type=str, default='Dahoas/full-hh-rlhf')

parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--num-test-prompt', type=int, default=3125)
parser.add_argument('--uncertainty-threshold', type=float, default=3.)

args = parser.parse_args()

rs = RewardSampling(llm_dir=args.llm_dir, rm_dir=args.rm_dir, seed=args.seed)
test_data = data_loader.QA_loader(args.data_dir, split='test', batch_size=1, head=args.num_test_prompt)

with autocast(dtype=torch.bfloat16):
    reward_full = torch.empty(0, dtype=torch.bfloat16)
    reward_partial = torch.empty(0, dtype=torch.bfloat16)
    num_subsentence, len_subsentence = [], []

    for _, response in tqdm(test_data):
        r_full = rs.rm_score(response).detach().cpu()
        reward_full = torch.cat([reward_full, r_full])

        # partition responses
        with torch.no_grad():
            tokens, mask = rs.from_text_to_token(response)
            full_logit = rs.from_token_to_full_logit(tokens, mask)
            u = uncertainty.entropy(full_logit).detach().cpu()[0]
        
        partition_mask = (u > args.uncertainty_threshold)
        response_partial, l = [], 0
        for i in range(3, len(partition_mask)):
            if partition_mask[i]:
                response_partial.append(rs.from_token_to_text(tokens[:,l:i])[0])
                l = i
        if l < len(partition_mask) - 1:
            response_partial.append(rs.from_token_to_text(tokens[:,l:])[0])
        r_partial = rs.rm_score(response_partial).detach().cpu()
        reward_partial = torch.cat([reward_partial, r_partial.mean().unsqueeze(0)])
        num_subsentence.append(len(response_partial))
        _len_sub = [len(r) for r in response_partial]
        len_subsentence.append(sum(_len_sub) / len(_len_sub))

    np.save('reward_full.npy', reward_full.to(torch.float32).numpy())
    np.save('reward_partial.npy', reward_partial.to(torch.float32).numpy())
    np.save('num_subsentence.npy', np.array(num_subsentence))
    np.save('len_subsentence.npy', np.array(len_subsentence))
