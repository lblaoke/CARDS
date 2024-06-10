import sys
sys.path.append('.')

import torch
from datasets import load_dataset

from reward_sampling import RewardSampling
import data_loader

from tqdm import tqdm
from time import time

rs = RewardSampling(llm_dir='argsearch/llama-7b-sft-float32', rm_dir='argsearch/llama-7b-rm-float32')

test_data = data_loader.QALoader('Dahoas/full-hh-rlhf', split='test', head=1000)
dataloader = torch.utils.data.DataLoader(test_data, batch_size=8, num_workers=0)

# RM
_time = time()
with torch.no_grad():
    for batch_idx, (prompt, _) in tqdm(enumerate(dataloader)):
        tokens, mask = rs.from_text_to_token(prompt)
        rs.from_token_to_reward(tokens, mask, None)
second = (time() - _time) / (batch_idx + 1)
print(f"RM: {second=} s per iteration")

# LLM
_time = time()
with torch.no_grad():
    for batch_idx, (prompt, _) in tqdm(enumerate(dataloader)):
        tokens, mask = rs.from_text_to_token(prompt)
        rs.from_token_to_logit(tokens, mask, None)
second = (time() - _time) / (batch_idx + 1)
print(f"LLM: {second=} s per iteration")

# RM
_time = time()
with torch.no_grad():
    for batch_idx, (prompt, _) in tqdm(enumerate(dataloader)):
        tokens, mask = rs.from_text_to_token(prompt)
        rs.from_token_to_reward(tokens, mask, None)
second = (time() - _time) / (batch_idx + 1)
print(f"RM: {second=} s per iteration")

# LLM
_time = time()
with torch.no_grad():
    for batch_idx, (prompt, _) in tqdm(enumerate(dataloader)):
        tokens, mask = rs.from_text_to_token(prompt)
        rs.from_token_to_logit(tokens, mask, None)
second = (time() - _time) / (batch_idx + 1)
print(f"LLM: {second=} s per iteration")
