import sys
sys.path.append(".")

import torch
from torch.cuda.amp import autocast
from datasets import load_dataset

from reward_sampling import RewardSampling
import data_loader

import argparse
import json
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str, default="grad_rs")
parser.add_argument("--save", type=str, default=None)

# Llama LLM choices
parser.add_argument("--llm-dir", type=str, default="argsearch/llama-7b-sft-float32")
# parser.add_argument("--llm-dir", type=str, default="meta-llama/Llama-2-13b-chat-hf")
# parser.add_argument("--llm-dir", type=str, default="ewqr2130/llama_ppo_1e6_new_tokenizerstep_8000")
# parser.add_argument("--llm-dir", type=str, default="AmberYifan/llama-7b-sft-DPO")

# Llama RM choices
parser.add_argument("--rm-dir", type=str, default="argsearch/llama-7b-rm-float32")
# parser.add_argument("--rm-dir", type=str, default="weqweasdas/hh_rlhf_rm_open_llama_3b")
# parser.add_argument("--rm-dir", type=str, default="miulab/llama2-7b-ultrafeedback-rm")

# Mistral LLM choices
# parser.add_argument("--llm-dir", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
# parser.add_argument("--llm-dir", type=str, default="renyiyu/mistral-7b-instruct-v0.2-bnb-4bit-ppo-v0")
# parser.add_argument("--llm-dir", type=str, default="AmberYifan/Mistral-7B-Instruct-v0.2-DPO")

# Mistral RM choices
# parser.add_argument("--rm-dir", type=str, default="Ray2333/reward-model-Mistral-7B-instruct-Unified-Feedback")
# parser.add_argument("--rm-dir", type=str, default="weqweasdas/RM-Mistral-7B")

# Datasets
parser.add_argument("--data-dir", type=str, default="Dahoas/full-hh-rlhf")
# parser.add_argument("--data-dir", type=str, default="nvidia/HelpSteer")
# parser.add_argument("--data-dir", type=str, default="PKU-Alignment/BeaverTails")

# Common Options
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--num-test-prompt", type=int, default=100)
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--max-new-token", type=int, default=128)

# CARDS Options
parser.add_argument("--entropy", type=float, default=3.0)
parser.add_argument("--reward", type=float, default=8.5) # for UF_loader: 10.0 (ARGS 9.32526); for Ray RM: 1.2 (ARGS 0.262109375).
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--beta", type=float, default=0.7)

# GradRS Options
parser.add_argument("--lr", type=float, default=0.1)

args = parser.parse_args()

print(f"\n{args.method=}")
print(f"{args.data_dir=}")
print(f"{args.llm_dir=}")
print(f"{args.rm_dir=}\n")

# init sampler
sampler = RewardSampling(access_token=None, llm_dir=args.llm_dir, rm_dir=args.rm_dir, seed=args.seed)
data = data_loader.QA_loader(args.data_dir, split="test", batch_size=args.batch_size, head=args.num_test_prompt)
# data = data_loader.UF_loader(batch_size=args.batch_size, head=args.num_test_prompt)

num_batch, total_reward, total_num_llm_call, total_num_rm_call = 0, 0, 0, 0

if args.save is not None:
    f = open(f"{args.save}.jsonl", 'w')

with autocast(dtype=torch.bfloat16, enabled=True):
    for prompt, _ in tqdm(data):
        if args.method == "grad_rs":
            response, reward, num_operation = sampler.grad_rs_generate(
                prompt,
                reward_threshold=args.reward,
                lr=args.lr,
                beta=args.beta,
            )
            num_llm_call, num_rm_call = num_operation["llm_call"], num_operation["rm_call"]

        elif args.method == "cards":
            response, (reward, num_llm_call, num_rm_call) = sampler.seg_rs_generate(
                prompt,
                option="soft",
                max_new_token=args.max_new_token,
                entropy_threshold=args.entropy,
                reward_threshold=args.reward,
                alpha=args.alpha,
                beta=args.beta
            )

        elif args.method == "rs":
            response, (reward, num_llm_call, num_rm_call) = sampler.rs_generate(
                prompt,
                max_new_token=args.max_new_token,
                reward_threshold=args.reward,
                beta=args.beta
            )

        elif args.method == "sentence_cards":
            response, (reward, num_llm_call, num_rm_call) = sampler.seg_sentence_rs_generate(
                prompt,
                option="soft",
                max_new_token=args.max_new_token,
                reward_threshold=args.reward
            )

        elif args.method == "bon":
            response, (reward, num_llm_call, num_rm_call) = sampler.bon_generate(prompt, max_new_token=args.max_new_token)

        elif args.method == "args":
            response, (reward, num_llm_call, num_rm_call) = sampler.token_bon_generate(prompt, max_new_token=args.max_new_token)

        elif args.method == "bolt":
            response, reward, num_operation = sampler.bolt_generate(
                prompt,
                reward_threshold=args.reward,
                beta=args.beta,
            )
            num_llm_call, num_rm_call = num_operation["llm_call"], num_operation["rm_call"]

        else:
            response, (reward, num_llm_call, num_rm_call) = sampler.generate(prompt, max_new_token=args.max_new_token)

        num_batch += 1
        total_reward += reward
        total_num_llm_call += num_llm_call
        total_num_rm_call += num_rm_call

        if args.save is not None:
            for i in range(len(prompt)):
                json.dump({"prompt": prompt[i], "response": response[i]}, f, ensure_ascii=False)
                f.write("\n")

if args.save is not None:
    f.close()

print(f"Average reward: {total_reward / num_batch}")
print(f"Average number of LLM calls: {total_num_llm_call / (num_batch * args.batch_size)}")
print(f"Average number of RM calls: {total_num_rm_call / (num_batch * args.batch_size)}")
