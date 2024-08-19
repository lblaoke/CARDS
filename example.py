from reward_sampling import RewardSampling

rs = RewardSampling(access_token=None, llm_dir='argsearch/llama-7b-sft-float32', rm_dir='argsearch/llama-7b-rm-float32')

# reward: batch_size x 1
reward = rs.rm_score(['How to write a paper?', 'How are you?'])
