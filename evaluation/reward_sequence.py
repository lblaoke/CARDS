import sys
sys.path.append('.')

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
parser.add_argument('--llm_dir', type=str, default='argsearch/llama-7b-sft-float32')
parser.add_argument('--rm_dir', type=str, default='argsearch/llama-7b-rm-float32')
parser.add_argument('--data_dir', type=str, default='Dahoas/full-hh-rlhf')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--text', type=str)

args = parser.parse_args()

rs = RewardSampling(llm_dir=args.llm_dir, rm_dir=args.rm_dir, seed=args.seed)

words, rewards = [], []

with autocast(dtype=torch.bfloat16):
    tokens, mask = rs.from_text_to_token(args.text)

    for i in range(1, tokens.shape[-1] + 1):
        r, _ = rs.from_token_to_reward(tokens[:, :i], mask=mask[:, :i])
        rewards.append(r.detach().cpu().item())

for i in range(tokens.shape[-1]):
    words.append(rs.from_token_to_text(tokens[:, i])[0])

print(words)
print(rewards)
