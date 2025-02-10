from reward_sampling import RewardSampling

# rs = RewardSampling(access_token=None, llm_dir="meta-llama/Llama-3.1-8B-Instruct", rm_dir="Skywork/Skywork-Reward-Llama-3.1-8B-v0.2")
rs = RewardSampling(llm_dir="argsearch/llama-7b-sft-float32", rm_dir="argsearch/llama-7b-rm-float32")
# rs = RewardSampling(llm_dir='argsearch/llama-7b-sft-float32', dpo_dir='AmberYifan/llama-7b-sft-DPO')

# reward: batch_size x 1
# print(rs.seg_sentence_rs_generate("Hi, ", max_new_token=8))
print(rs.seg_punctuation_rs_generate('Human: Ivory soap seems to have been around forever. Do you know much about its history? Assistant:', max_new_token=128))

# import torch
# token = torch.tensor([[88,2,3,4,5,6,7,8,9,10,11]]).cuda()
# # mask = torch.tensor([[0,0,0,1,1,1]]).cuda()
# r = rs.from_token_to_weighted_implicit_reward(token, prompt_len=4)[0]
# print(r.shape, r)
