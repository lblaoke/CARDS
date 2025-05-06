import sys
sys.path.append('.')

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from datasets import load_dataset

from reward_sampling import RewardSampling
import data_loader
from tools import uncertainty

import argparse
import json
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--llm_dir', type=str, default='facebook/opt-125m')
parser.add_argument('--rm_dir', type=str, default='argsearch/llama-7b-rm-float32')
parser.add_argument('--data_dir', type=str, default='Dahoas/full-hh-rlhf')
parser.add_argument('--batch_size', type=int, default=1)

args = parser.parse_args()

rs = RewardSampling(llm_dir=args.llm_dir, rm_dir=args.rm_dir)
data = data_loader.PRCR_loader(args, rs.tokenizer, head=100, max_length=512)

with autocast(dtype=torch.bfloat16):
    likelihood = []

    for row in tqdm(data):
        token, mask = rs.from_text_to_token([row['prompt'][0] + ' ' + row['chosen'][0]])
        token, mask = torch.tensor(token, device=rs.LLM.device), torch.tensor(mask, device=rs.LLM.device)

        logits, _ = rs.from_token_to_full_logit(token, mask)
        dist = F.softmax(logits, dim=-1)
        prob = dist[:, :-1, :].gather(-1, token[:, 1:].unsqueeze(-1)).squeeze(-1).detach()
        likelihood.append(prob.mean().item())

print('chosen likelihood:', sum(likelihood)/len(likelihood))

with autocast(dtype=torch.bfloat16):
    likelihood = []

    for row in tqdm(data):
        token, mask = rs.from_text_to_token([row['prompt'][0] + ' ' + row['rejected'][0]])
        token, mask = torch.tensor(token, device=rs.LLM.device), torch.tensor(mask, device=rs.LLM.device)

        logits, _ = rs.from_token_to_full_logit(token, mask)
        dist = F.softmax(logits, dim=-1)
        prob = dist[:, :-1, :].gather(-1, token[:, 1:].unsqueeze(-1)).squeeze(-1).detach()
        likelihood.append(prob.mean().item())

print('rejected likelihood:', sum(likelihood)/len(likelihood))

# implicit reward ACC
with autocast(dtype=torch.bfloat16):
    correct = []

    for row in tqdm(data):
        token, mask = rs.from_text_to_token([row['prompt'][0] + ' ' + row['chosen'][0]])
        token, mask = torch.tensor(token, device=rs.LLM.device), torch.tensor(mask, device=rs.LLM.device)
        r_w = rs.from_token_to_implicit_reward(token, mask)[0]

        token, mask = rs.from_text_to_token([row['prompt'][0] + ' ' + row['rejected'][0]])
        token, mask = torch.tensor(token, device=rs.LLM.device), torch.tensor(mask, device=rs.LLM.device)
        r_l = rs.from_token_to_implicit_reward(token, mask)[0]

        correct.append(r_w.item() > r_l.item())

print('implicit reward ACC:', sum(correct)/len(correct))

output = []
with autocast(dtype=torch.bfloat16):
    likelihood = []

    for row in tqdm(data):
        response, _ = rs.generate(tokens=row['input_ids'], mask=row['attention_mask'], max_new_token=128)
        for r in response:
            output.append({'output': r})

        token, mask = rs.from_text_to_token(response)
        token, mask = torch.tensor(token, device=rs.LLM.device), torch.tensor(mask, device=rs.LLM.device)

        logits, _ = rs.from_token_to_full_logit(token, mask)
        dist = F.softmax(logits, dim=-1)
        prob = dist[:, :-1, :].gather(-1, token[:, 1:].unsqueeze(-1)).squeeze(-1).detach()
        likelihood.append(prob.mean().item())

print('generated likelihood:', sum(likelihood)/len(likelihood))

rs.unload_all()
rs.load_rm()
total_reward = 0.

with autocast(dtype=torch.bfloat16, enabled=True):
    for row in output:
        reward = rs.get_reward([row['output']])
        total_reward += reward

print(f'Average reward: {total_reward / len(output)}')
