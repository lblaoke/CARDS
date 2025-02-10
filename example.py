from reward_sampling import RewardSampling

# rs = RewardSampling(access_token=None, llm_dir="meta-llama/Llama-3.1-8B-Instruct", rm_dir="Skywork/Skywork-Reward-Llama-3.1-8B-v0.2")
rs = RewardSampling(llm_dir="argsearch/llama-7b-sft-float32", rm_dir="argsearch/llama-7b-rm-float32")
# rs = RewardSampling(llm_dir='argsearch/llama-7b-sft-float32', dpo_dir='AmberYifan/llama-7b-sft-DPO')

# reward: batch_size x 1
# print(rs.seg_sentence_rs_generate("Hi, ", max_new_token=8))
# print(rs.seg_fix_rs_generate('Hi, ', max_new_token=8))

# import torch
# mask = torch.tensor([[0,0,0,1,1,1]]).cuda()
with open('f1.txt') as f:
    f1 = f.read()
with open('f2.txt') as f:
    f2 = f.read()
with open('r1.txt') as f:
    r1 = f.read()
with open('r2.txt') as f:
    r2 = f.read()
r = rs.from_token_to_self_reward(f1, f2, r1, r2, 'Hi. I am a man.')[0]
print(r)
