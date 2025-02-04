from reward_sampling import RewardSampling

# rs = RewardSampling(access_token=None, llm_dir="meta-llama/Llama-3.1-8B-Instruct", rm_dir="Skywork/Skywork-Reward-Llama-3.1-8B-v0.2")
rs = RewardSampling(llm_dir="argsearch/llama-7b-sft-float32", rm_dir="argsearch/llama-7b-rm-float32")

# reward: batch_size x 1
print(rs.grad_rs_generate("Hi, ", max_new_token=8))
