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
# parser.add_argument('--rm-dir', type=str, default='argsearch/llama-7b-rm-float32')
# parser.add_argument('--llm-dir', type=str, default='mistralai/Mistral-7B-Instruct-v0.2')
# parser.add_argument('--rm-dir', type=str, default='weqweasdas/RM-Mistral-7B')

parser.add_argument('--dpo_dir', type=str, default='AmberYifan/llama-7b-sft-DPO')

parser.add_argument('--data-dir', type=str, default='Dahoas/full-hh-rlhf')

parser.add_argument('--seed', type=int, default=1)

args = parser.parse_args()

rs = RewardSampling(llm_dir=args.llm_dir, dpo_dir=args.dpo_dir, seed=args.seed)
data = data_loader.fast_QA_loader(args.data_dir, split='train', batch_size=1, head=1000)

with autocast(dtype=torch.bfloat16):
    reward = []

    for _, response in tqdm(data):
        r = rs.rm_score(response).detach().cpu()
        reward.append(r.item())

print(sum(reward)/len(reward))

# np.save('mistral_reward_dist.npy', torch.cat(reward).to(torch.float32).numpy())
