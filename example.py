from reward_sampling import RewardSampling

rs = RewardSampling(access_token=None, llm_dir='meta-llama/Llama-2-13b-chat-hf', rm_dir='vincentmin/llama-2-13b-reward-oasst1')

# reward: batch_size x 1
reward = rs.rm_score(['How to write a paper?', 'How are you?'])
