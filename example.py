from reward_sampling import RewardSampling

# rs = RewardSampling(access_token=None, llm_dir="meta-llama/Llama-3.1-8B-Instruct", rm_dir="Skywork/Skywork-Reward-Llama-3.1-8B-v0.2")
# rs = RewardSampling(llm_dir="argsearch/llama-7b-sft-float32", rm_dir="argsearch/llama-7b-rm-float32")
# rs = RewardSampling(llm_dir='argsearch/llama-7b-sft-float32', dpo_dir='AmberYifan/llama-7b-sft-DPO')

# rs = RewardSampling(llm_dir='meta-llama/Meta-Llama-3-8B-Instruct', rm_dir='Ray2333/GRM-Llama3-8B-rewardmodel-ft', draft_dir='turboderp/Qwama-0.5B-Instruct')
rs = RewardSampling(llm_dir='lblaoke/qwama-0.5b-skywork-pref-dpo-trl-v2', rm_dir='turboderp/Qwama-0.5B-RewardModel')

a, b = rs.from_text_to_token("Hello, how are you?")
print(a, type(a))
print(b, type(b))