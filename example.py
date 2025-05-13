from reward_sampling import RewardSampling

sampler = RewardSampling(llm_dir="meta-llama/Meta-Llama-3-8B-Instruct", draft_dir="turboderp/Qwama-0.5B-Instruct")
# sampler = RewardSampling(llm_dir="turboderp/Qwama-0.5B-Instruct")

# sampler = RewardSampling(llm_dir="facebook/opt-6.7b", draft_dir="facebook/opt-125m")

ret = sampler.sd_generate(["Can you explain the difference between a cat and a dog?"], max_new_token=64)

print(ret[0][0])

print("==="*20)

ret = sampler.generate(["Can you explain the difference between a cat and a dog?"], max_new_token=64)

print(ret[0][0])
