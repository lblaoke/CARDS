import sys
sys.path.append('.')

import torch
from torch.cuda.amp import autocast

from reward_sampling import RewardSampling
import data_loader

import argparse
import json
import yaml
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, required=True)
parser.add_argument('--save', type=str)
parser.add_argument('--num_test_prompt', type=int)
args = parser.parse_args()

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

for key, value in config.items():
    if not hasattr(args, key) or getattr(args, key) is None:
        setattr(args, key, value)

# parser.add_argument('--method', type=str, default='sd')
# parser.add_argument('--save', type=str, default=None)
# parser.add_argument('--save_fmt', type=str, default='jsonl')

# Llama LLM choices
# parser.add_argument('--llm_dir', type=str, default='argsearch/llama-7b-sft-float32')
# parser.add_argument('--llm_dir', type=str, default='meta-llama/Llama-2-13b-chat-hf')
# parser.add_argument('--llm_dir', type=str, default='ContextualAI/archangel_ppo_llama7b')
# parser.add_argument('--dpo_dir', type=str, default='AmberYifan/llama-7b-sft-DPO')

# parser.add_argument('--llm_dir', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct')
# parser.add_argument('--draft_dir', type=str, default='turboderp/Qwama-0.5B-Instruct')
# parser.add_argument('--draft_dir', type=str, default='lblaoke/qwama-0.5b-skywork-pref-dpo-trl-v2')
# parser.add_argument('--dpo_dir', type=str, default='allenai/llama-3-tulu-2-dpo-8b')

# Llama RM choices
# parser.add_argument('--rm_dir', type=str, default='argsearch/llama-7b-rm-float32')
# parser.add_argument('--rm_dir', type=str, default='weqweasdas/hh_rlhf_rm_open_llama_3b')
# parser.add_argument('--rm_dir', type=str, default='miulab/llama2-7b-ultrafeedback-rm')
# parser.add_argument('--rm_dir', type=str, default='Ray2333/GRM-Llama3-8B-rewardmodel-ft')

# Mistral LLM choices
# parser.add_argument('--llm_dir', type=str, default='mistralai/Mistral-7B-Instruct-v0.2')
# parser.add_argument('--llm_dir', type=str, default='renyiyu/mistral-7b-instruct-v0.2-bnb-4bit-ppo-v0')
# parser.add_argument('--llm_dir', type=str, default='AmberYifan/Mistral-7B-Instruct-v0.2-DPO')

# Mistral RM choices
# parser.add_argument('--rm_dir', type=str, default='Ray2333/reward-model-Mistral-7B-instruct-Unified-Feedback')
# parser.add_argument('--rm_dir', type=str, default='weqweasdas/RM-Mistral-7B')

# Datasets
# parser.add_argument('--data_dir', type=str, default='Dahoas/full-hh-rlhf')
# parser.add_argument('--data_dir', type=str, default='nvidia/HelpSteer')
# parser.add_argument('--data_dir', type=str, default='PKU-Alignment/BeaverTails')
# parser.add_argument('--data_dir', type=str, default='alpaca_eval.json')
# parser.add_argument('--data_dir', type=str, default='walledai/AdvBench')
# parser.add_argument('--data_dir', type=str, default='saferlhf.jsonl')
# parser.add_argument('--data_dir', type=str, default='Skywork/Skywork-Reward-Preference-80K-v0.1')

# Common Options
# parser.add_argument('--seed', type=int, default=1)
# parser.add_argument('--num_test_prompt', type=int, default=None)
# parser.add_argument('--batch_size', type=int, default=1)
# parser.add_argument('--max_new_token', type=int, default=128)

# CARDS Options
# parser.add_argument('--entropy', type=float, default=2.0)
# parser.add_argument('--reward', type=float, default=9.0) # for UF_loader: 10.0 (ARGS 9.32526); for Ray RM: 1.2 (ARGS 0.262109375).
# parser.add_argument('--alpha', type=float, default=0.5)
# parser.add_argument('--beta', type=float, default=0.7) # 0.7 for CARDS, 0.8 for standard SD

# # GradRS Options
# parser.add_argument('--lr', type=float, default=0.1)

print(f'\n{args.method=}')
print(f'{args.data_dir=}')
print(f'{args.llm_dir=}')
print(f'{args.rm_dir=}\n')

# init sampler
sampler_kwargs = {
    'access_token': None,
    'llm_dir': args.llm_dir,
    'rm_dir': args.rm_dir,
    'draft_dir': args.draft_dir if hasattr(args, 'draft_dir') else None,
    'dpo_dir': args.dpo_dir if hasattr(args, 'dpo_dir') else None,
    'seed': args.seed,
}
sampler = RewardSampling(**sampler_kwargs)

data = data_loader.PRCR_loader(args, sampler.tokenizer, head=args.num_test_prompt, max_length=args.max_length)

num_samples, total_num_llm_call, total_num_rm_call = 0, 0, 0
output = []

with autocast(dtype=torch.bfloat16, enabled=True):
    for row in tqdm(data):
        prompt, tokens, mask = row['prompt'], row['input_ids'], row['attention_mask']

        if args.method == 'sd':
            response, (num_llm_call, num_rm_call) = sampler.sd_generate(
                prompt,
                beta=args.temperature,
                max_new_token=args.max_new_token,
            )

        elif args.method == 'grad_rs':
            assert args.batch_size == 1, 'grad_rs does not support batch_size > 1'
            response, num_operation = sampler.grad_rs_generate(
                prompt,
                reward_threshold=args.reward,
                lr=args.lr,
                beta=args.temperature,
            )
            num_llm_call, num_rm_call = num_operation['llm_call'], num_operation['rm_call']

        elif args.method == 'cards':
            assert args.batch_size == 1, 'cards does not support batch_size > 1'
            response, (num_llm_call, num_rm_call) = sampler.seg_rs_generate(
                prompt,
                option='soft',
                max_new_token=args.max_new_token,
                entropy_threshold=args.entropy,
                reward_threshold=args.reward,
                alpha=args.alpha,
                beta=args.temperature
            )

        elif args.method == 'rs':
            assert args.batch_size == 1, 'rs does not support batch_size > 1'
            response, (num_llm_call, num_rm_call) = sampler.rs_generate(
                prompt,
                max_new_token=args.max_new_token,
                reward_threshold=args.reward,
                beta=args.temperature
            )

        elif args.method == 'punctuation_cards':
            assert args.batch_size == 1, 'punctuation_cards does not support batch_size > 1'
            response, (num_llm_call, num_rm_call) = sampler.seg_punctuation_rs_generate(
                prompt,
                max_new_token=args.max_new_token,
                reward_threshold=args.reward
            )

        elif args.method == 'treebon': # max_r = 4.0
            assert args.batch_size == 1, 'treebon does not support batch_size > 1'
            response, (num_llm_call, num_rm_call) = sampler.seg_fix_rs_generate(
                prompt,
                method='treebon',
                max_new_token=args.max_new_token,
                reward_threshold=4.0#args.reward
            )

        elif args.method == 'rain': # max_r = 1.5
            assert args.batch_size == 1, 'rain does not support batch_size > 1'
            response, (num_llm_call, num_rm_call) = sampler.seg_fix_rs_generate(
                prompt,
                method='rain',
                max_new_token=args.max_new_token,
                reward_threshold=1.5#args.reward
            )

        elif args.method == 'bon':
            assert args.batch_size == 1, 'bon does not support batch_size > 1'
            response, (num_llm_call, num_rm_call) = sampler.bon_generate(prompt, max_new_token=args.max_new_token)

        elif args.method == 'args':
            response, (num_llm_call, num_rm_call) = sampler.token_bon_generate(prompt, max_new_token=args.max_new_token)

        elif args.method == 'bolt':
            assert args.batch_size == 1, 'bolt does not support batch_size > 1'
            response, num_operation = sampler.bolt_generate(
                prompt,
                reward_threshold=args.reward,
                beta=args.temperature,
            )
            num_llm_call, num_rm_call = num_operation['llm_call'], num_operation['rm_call']

        else:
            response, (num_llm_call, num_rm_call) = sampler.generate(
                tokens=tokens,
                mask=mask,
                top_k=args.top_k,
                beta=args.temperature,
                max_new_token=args.max_new_token,
            )

        num_samples += len(prompt)
        total_num_llm_call += num_llm_call
        total_num_rm_call += num_rm_call

        for i in range(len(prompt)):
            output.append({'instruction': prompt[i], 'output': response[i]})

# save generation results
if args.save is not None:
    if args.save_fmt == 'jsonl':
        with open(f'{args.save}.jsonl', 'w') as f:
            for row in output:
                json.dump(row, f, ensure_ascii=False)
                f.write('\n')
    elif args.save_fmt == 'json':
        with open(f'{args.save}.json', 'w') as f:
            json.dump(output, f, ensure_ascii=False)
    else:
        assert False, f'Unknown save_format: {args.save_fmt}'

# reward evaluation
sampler.unload_all()
sampler.load_rm()
total_reward = 0.

with autocast(dtype=torch.bfloat16, enabled=True):
    for row in output:
        reward = sampler.get_reward([row['output']])
        total_reward += reward

print(f'Average reward: {total_reward / num_samples}')
print(f'Average number of LLM calls: {total_num_llm_call / num_samples}')
print(f'Average number of RM calls: {total_num_rm_call / num_samples}')
