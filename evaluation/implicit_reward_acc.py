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
# parser.add_argument('--llm_dir', type=str, default='lblaoke/qwama-0.5b-skywork-pref-dpo-trl-v2')
parser.add_argument('--llm_dir', type=str, default='turboderp/Qwama-0.5B-Instruct')
parser.add_argument('--rm_dir', type=str, default='Ray2333/GRM-Llama3-8B-rewardmodel-ft')

args = parser.parse_args()

rs = RewardSampling(llm_dir=args.llm_dir, rm_dir=args.rm_dir)

data = data_loader.Pref_loader('Skywork/Skywork-Reward-Preference-80K-v0.1', split='train', batch_size=1, head=100, data_fmt='skywork_pref')

with autocast(dtype=torch.bfloat16):
    likelihood = []

    for prompt, response in tqdm(data):
        token, mask = rs.from_text_to_token('User:\n' + prompt[0] + '\n\nAssistant:\n' + response[0])
        # response, _ = rs.generate('User:\n' + prompt[0] + '\n\nAssistant:\n', max_new_token=128)
        # token, mask = rs.from_text_to_token(response)

        logits, _ = rs.from_token_to_full_logit(token, mask)
        dist = F.softmax(logits, dim=-1)
        prob = dist[:, :-1, :].gather(-1, token[:, 1:].unsqueeze(-1)).squeeze(-1).detach()
        likelihood.append(prob.mean().item())

print(sum(likelihood)/len(likelihood))

# np.save('mistral_reward_dist.npy', torch.cat(likelihood).to(torch.float32).numpy())
