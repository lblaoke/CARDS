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
parser.add_argument('--llm-dir', type=str, default='argsearch/llama-7b-sft-float32')
parser.add_argument('--data-dir', type=str, default='Dahoas/full-hh-rlhf')

parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--num-test-prompt', type=int, default=100) # 3125
parser.add_argument('--batch-size', type=int, default=1)

parser.add_argument('--entropy', type=float)

args = parser.parse_args()

rs = RewardSampling(access_token='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', llm_dir=args.llm_dir, seed=args.seed)
test_data = data_loader.QA_loader(args.data_dir, split='test', batch_size=args.batch_size, head=args.num_test_prompt)

avg_len, avg_num = [], []

with autocast(dtype=torch.bfloat16):
    for prompt, response in tqdm(test_data):
        tokens, mask = rs.from_text_to_token(response)
        logits = rs.from_token_to_full_logit(tokens, mask)
        u =  uncertainty.entropy(logits[0]).cpu()
        start_point = (u >= args.entropy)

        avg_len.append(len(u))
        avg_num.append(start_point.count_nonzero().item() + 1)

avg_segment_len = [avg_len[i] / avg_num[i] for i in range(len(avg_len))]

print(avg_len)
print(avg_num)
print(avg_segment_len)

import numpy as np
np.save('resposne_len.npy', np.array(avg_len))
np.save('num_segment.npy', np.array(avg_num))
np.save('segment_len.npy', np.array(avg_segment_len))

# print('Average segment length:', sum(avg_segment_len) / len(avg_segment_len))
# print('Average number of segments:', sum(avg_num) / len(avg_num))
# print('Average number of segment length:', sum(avg_len) / len(avg_len))
