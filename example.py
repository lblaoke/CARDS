from reward_sampling import RewardSampling

# rs = RewardSampling(access_token=None, llm_dir="meta-llama/Llama-3.1-8B-Instruct", rm_dir="Skywork/Skywork-Reward-Llama-3.1-8B-v0.2")
# rs = RewardSampling(llm_dir="argsearch/llama-7b-sft-float32", rm_dir="argsearch/llama-7b-rm-float32")
# rs = RewardSampling(llm_dir='argsearch/llama-7b-sft-float32', dpo_dir='AmberYifan/llama-7b-sft-DPO')

rs = RewardSampling(llm_dir='meta-llama/Meta-Llama-3-8B-Instruct', rm_dir='Ray2333/GRM-Llama3-8B-rewardmodel-ft', draft_dir='turboderp/Qwama-0.5B-Instruct')

# reward: batch_size x 1
response, _ = rs.generate("How did US states get their names?", max_new_token=64)
