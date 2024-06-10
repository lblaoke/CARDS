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
parser.add_argument('--num-test-prompt', type=int, default=10)

args = parser.parse_args()

rs = RewardSampling(llm_dir=args.llm_dir, rm_dir=args.rm_dir, seed=args.seed)
test_data = data_loader.HH_loader(args.data_dir, split='test', batch_size=1, head=args.num_test_prompt)

with autocast(dtype=torch.bfloat16):
    idx = 0

    for _, chosen, rejected in tqdm(test_data):
        # uncertainty
        with torch.no_grad():
            tokens_c, _ = rs.from_text_to_token(chosen)
            full_logit = rs.from_token_to_full_logit(tokens_c)
            u_c = uncertainty.entropy(full_logit).detach().cpu()[0][:-1]

            tokens_r, _ = rs.from_text_to_token(rejected)
            full_logit = rs.from_token_to_full_logit(tokens_r)
            u_r = uncertainty.entropy(full_logit).detach().cpu()[0][:-1]

        # reward
        r_c, r_r = [], []
        with torch.no_grad():
            for i in range(1, tokens_c.shape[1]):
                r_ci, _ = rs.from_token_to_reward(tokens_c[:,:i])
                r_c.append(r_ci.detach().cpu().item())
            for i in range(1, tokens_r.shape[1]):
                r_ri, _ = rs.from_token_to_reward(tokens_r[:,:i])
                r_r.append(r_ri.detach().cpu().item())

        # save
        assert (len(u_c) == len(r_c)) and (len(u_r) == len(r_r)), f'{len(u_c)} {len(u_r)} {len(r_c)} {len(r_r)}'

        np.save(f'token_check_tmp/u_c{idx}.npy', u_c.to(torch.float32).numpy())
        np.save(f'token_check_tmp/u_r{idx}.npy', u_r.to(torch.float32).numpy())
        np.save(f'token_check_tmp/r_c{idx}.npy', np.array(r_c))
        np.save(f'token_check_tmp/r_r{idx}.npy', np.array(r_r))
        
        idx += 1
