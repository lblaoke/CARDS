import sys
sys.path.append('.')

import numpy as np
import torch
from torch.cuda.amp import autocast

from reward_sampling import RewardSampling

import argparse
import json
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--llm-dir', type=str, default='argsearch/llama-7b-sft-float32')
parser.add_argument('--rm-dir', type=str, default='argsearch/llama-7b-rm-float32')
# parser.add_argument('--llm-dir', type=str, default='mistralai/Mistral-7B-Instruct-v0.2')
# parser.add_argument('--rm-dir', type=str, default='weqweasdas/RM-Mistral-7B')
parser.add_argument('--result-dir', type=str, default='evaluation/hh_rlhf_output/llama_7b_args100.jsonl')
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()

rs = RewardSampling(llm_dir=args.llm_dir, rm_dir=args.rm_dir, seed=args.seed)

def load_responses(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            json_data = json.loads(line)
            data.append(json_data['response'])
    return data

data = load_responses(args.result_dir)

with autocast(dtype=torch.bfloat16):
    idx, rewards = 0, 0
    for response in tqdm(data):
        reward = rs.rm_score(response).detach().item()
        rewards += reward
        idx += 1

print(f"Average reward: {rewards / idx}")
