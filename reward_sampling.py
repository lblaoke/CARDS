import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, LlamaForCausalLM, LlamaForSequenceClassification

from tqdm import tqdm
import random
from math import *

from base_class import BaseRewardSampling
from tools import uncertainty

class RewardSampling(BaseRewardSampling):
    def __init__(
            self                    ,
            llm_dir     : str       ,
            rm_dir      : str=None  ,
            cache_dir   : str=None  ,
            access_token: str=None  ,

            seed        : int=1     ,
            fp_bit      : int=16    ,
            device_map  : str='auto'
        ):
        super(RewardSampling, self).__init__(seed=seed)

        assert fp_bit in {4, 8, 16, 32}, 'fp_bit must be one of {4, 8, 16, 32}!'

        print('==> Loading tokenizer...')
        self.tokenizer = AutoTokenizer.from_pretrained(llm_dir, token=access_token, cache_dir=cache_dir)

        print('==> Loading LLM...')
        self.LLM = AutoModelForCausalLM.from_pretrained(
            llm_dir                                                                     ,
            cache_dir           = cache_dir                                             ,
            token               = access_token                                          ,
            torch_dtype         = torch.bfloat16 if (fp_bit == 16) else torch.float32   ,
            load_in_8bit        = (fp_bit == 8)                                         ,
            load_in_4bit        = (fp_bit == 4)                                         ,
            device_map          = device_map                                            ,
            attn_implementation = 'flash_attention_2'
        )

        if rm_dir is not None:
            print('==> Loading RM...')
            self.RM = AutoModelForSequenceClassification.from_pretrained(
                rm_dir                                                                      ,
                cache_dir           = cache_dir                                             ,
                token               = access_token                                          ,
                num_labels          = 1                                                     ,
                torch_dtype         = torch.bfloat16 if (fp_bit == 16) else torch.float32   ,
                load_in_8bit        = (fp_bit == 8)                                         ,
                load_in_4bit        = (fp_bit == 4)                                         ,
                device_map          = device_map                                            ,
                attn_implementation = 'flash_attention_2'
            )

        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '</s>'})

            self.LLM.resize_token_embeddings(len(self.tokenizer))
            self.LLM.eval()

            if self.RM is not None:
                self.RM.resize_token_embeddings(len(self.tokenizer))

    # Vanilla LLM
    @torch.no_grad()
    def generate(self, prompt, max_new_token:int=128):
        tokens, mask = self.from_text_to_token(prompt)
        num_llm_call, num_rm_call = 0, 0
        llm_cache = None

        for _ in range(max_new_token):
            logits, llm_cache = self.from_token_to_logit(tokens, mask, llm_cache)
            num_llm_call += len(tokens)
            selected_token = self.from_logit_to_token(logits)

            tokens = torch.cat([tokens, selected_token], dim=-1)
            mask = torch.cat([mask, torch.ones_like(selected_token)], dim=-1)

        reward, _ = self.from_token_to_reward(tokens, mask)
        reward = reward.mean().item()
        return self.from_token_to_text(tokens), (reward, num_llm_call, num_rm_call)

    # ARGS: Alignment as Reward-Guided Search
    @torch.no_grad()
    def token_bon_generate(self, prompt, args_weight:float=1.5, topk:int=20, max_new_token:int=128):
        tokens, mask = self.from_text_to_token(prompt)
        num_llm_call, num_rm_call = 0, 0
        llm_cache, rm_cache = None, None

        for _ in range(max_new_token):
            logits, llm_cache = self.from_token_to_logit(tokens, mask, llm_cache)
            num_llm_call += len(tokens)
            val, idx = torch.topk(logits, k=topk, dim=-1)

            # reweight logits with rewards
            stacked_tokens = tokens.unsqueeze(1).repeat(1, topk, 1)
            stacked_tokens = torch.cat([stacked_tokens, idx.unsqueeze(-1)], dim=-1)
            stacked_tokens = stacked_tokens.view(-1, stacked_tokens.shape[-1])
            stacked_mask = mask.unsqueeze(1).repeat(1, topk, 1)
            stacked_mask = torch.cat([stacked_mask, torch.ones_like(idx.unsqueeze(-1))], dim=-1)
            stacked_mask = stacked_mask.view(-1, stacked_mask.shape[-1])

            reward, rm_cache = self.from_token_to_reward(stacked_tokens, stacked_mask, rm_cache)
            num_rm_call += len(tokens) * topk
            reward = reward.view(tokens.shape[0], topk)

            score = val + args_weight * reward
            selected_idx = torch.argmax(score, dim=-1, keepdim=True)
            selected_token = torch.gather(idx, -1, selected_idx)

            tokens = torch.cat([tokens, selected_token], dim=-1)
            mask = torch.cat([mask, torch.ones_like(selected_token)], dim=-1)

        reward, _ = self.from_token_to_reward(tokens, mask, rm_cache)
        reward = reward.mean().item()
        return self.from_token_to_text(tokens), (reward, num_llm_call, num_rm_call)

    # CARDS: Cascade Reward Sampling for Efficient Decoding-Time Alignment
    @torch.no_grad()
    def seg_rs_generate(
        self                                ,
        prompt                              ,
        option              : str   = 'soft',
        entropy_threshold   : float = 3.0   ,
        reward_threshold    : float = 8.5   ,
        alpha               : float = 0.5   ,
        beta                : float = 0.7   ,
        topk                : int   = 40    ,
        max_new_token       : int   = 128   ,
        debug               : bool  = False
    ):
        tokens, mask = self.from_text_to_token(prompt)
        len_prompt = tokens.shape[1]
        best_candidate, best_candidate_mask, best_reward = None, None, -1e34
        num_regeneration, num_llm_call, num_rm_call = 0, 0, 0
        llm_cache, rm_cache = None, None

        reward0, rm_cache = self.from_token_to_reward(tokens, mask, rm_cache)
        reward0 = (1 - alpha) * reward0.mean().item() + alpha * reward_threshold
        if debug: print(f'{reward0=:.2f}')

        def accept_check(reward, candidate):
            threshold = reward0 + (candidate.shape[1] - len_prompt) * (reward_threshold - reward0) / max_new_token
            threshold = min(threshold, reward_threshold)
            if debug: print(f'{reward=:.2f}, {threshold=:.2f}')

            if option == 'hard':
                return reward >= threshold
            elif option == 'soft':
                return random.uniform(0, 1) < min(1., exp((reward - threshold) / beta))
            else:
                assert False, 'Invalid reward sampling option!'

        while tokens.shape[1] - len_prompt < max_new_token:

            # sample a new candidate
            candidate = tokens.clone()
            candidate_mask = mask.clone()

            while candidate.shape[1] - tokens.shape[1] < 64:
                logits, llm_cache = self.from_token_to_logit(candidate, candidate_mask, llm_cache)
                num_llm_call += len(tokens)

                if candidate.shape[1] - tokens.shape[1] >= 4 and uncertainty.entropy(logits).mean().item() >= entropy_threshold:
                    del logits
                    break
                
                selected_token = self.from_logit_to_token(logits, top_k=topk, temperature=beta)
                candidate = torch.cat([candidate, selected_token], dim=-1)
                candidate_mask = torch.cat([candidate_mask, torch.ones_like(selected_token)], dim=-1)
            if debug: print(f'Segment length = {candidate.shape[1] - tokens.shape[1]}')

            # evaluate the candidate
            reward, rm_cache = self.from_token_to_reward(candidate, candidate_mask, rm_cache)
            num_rm_call += len(tokens)
            reward = reward.mean().item()

            if reward > best_reward:
                del best_candidate, best_candidate_mask
                best_candidate, best_candidate_mask, best_reward = candidate.clone(), candidate_mask.clone(), reward

            # accept/reject the candidate
            if num_regeneration >= 20:
                del tokens, mask, candidate, candidate_mask
                best_reward = -1e34
                tokens, mask = best_candidate, best_candidate_mask
                num_regeneration = 0
            elif accept_check(reward, candidate):
                del tokens, mask, best_candidate, best_candidate_mask
                best_candidate, best_candidate_mask, best_reward = None, None, -1e34
                tokens, mask = candidate, candidate_mask
                num_regeneration = 0
            else:
                del candidate, candidate_mask
                num_regeneration += 1
                if debug: print(f'Rejected {num_regeneration} times!')

        return self.from_token_to_text(tokens), (reward, num_llm_call, num_rm_call)

    # Item-level rejection sampling
    @torch.no_grad()
    def rs_generate(
        self                                ,
        prompt                              ,
        reward_threshold    : float = 8.5   ,
        beta                : float = 0.7   ,
        topk                : int   = 40    ,
        max_new_token       : int   = 128
    ):
        tokens, mask = self.from_text_to_token(prompt)
        len_prompt = tokens.shape[1]
        best_candidate, best_candidate_mask, best_reward = None, None, -1e34
        num_regeneration, num_llm_call, num_rm_call = 0, 0, 0
        llm_cache, rm_cache = None, None

        while tokens.shape[1] - len_prompt < max_new_token:

            # sample a new candidate
            candidate = tokens.clone()
            candidate_mask = mask.clone()

            while candidate.shape[1] - tokens.shape[1] < max_new_token:
                logits, llm_cache = self.from_token_to_logit(candidate, candidate_mask, llm_cache)
                num_llm_call += len(tokens)
                
                selected_token = self.from_logit_to_token(logits, top_k=topk, temperature=beta)
                candidate = torch.cat([candidate, selected_token], dim=-1)
                candidate_mask = torch.cat([candidate_mask, torch.ones_like(selected_token)], dim=-1)

            # evaluate the candidate
            reward, rm_cache = self.from_token_to_reward(candidate, candidate_mask, rm_cache)
            num_rm_call += len(tokens)
            reward = reward.mean().item()

            if reward > best_reward:
                del best_candidate, best_candidate_mask
                best_candidate, best_candidate_mask, best_reward = candidate.clone(), candidate_mask.clone(), reward

            # accept/reject the candidate
            if num_regeneration >= 200:
                del tokens, mask, candidate, candidate_mask
                best_reward = -1e34
                tokens, mask = best_candidate, best_candidate_mask
                num_regeneration = 0
            elif random.uniform(0, 1) < min(1., exp((reward - reward_threshold) / beta)):
                del tokens, mask, best_candidate, best_candidate_mask
                best_candidate, best_candidate_mask, best_reward = None, None, -1e34
                tokens, mask = candidate, candidate_mask
                num_regeneration = 0
            else:
                del candidate, candidate_mask
                num_regeneration += 1

        return self.from_token_to_text(tokens), (reward, num_llm_call, num_rm_call)

    # Sentence-level rejection sampling
    @torch.no_grad()
    def seg_sentence_rs_generate(
        self                                ,
        prompt                              ,
        option              : str   = 'soft',
        reward_threshold    : float = 8.5   ,
        alpha               : float = 0.5   ,
        beta                : float = 0.7   ,
        topk                : int   = 40   ,
        max_new_token       : int   = 128   ,
        debug               : bool  = False
    ):
        tokens, mask = self.from_text_to_token(prompt)
        len_prompt = tokens.shape[1]
        best_candidate, best_candidate_mask, best_reward = None, None, -1e34
        num_regeneration, num_llm_call, num_rm_call = 0, 0, 0
        llm_cache, rm_cache = None, None

        reward0, rm_cache = self.from_token_to_reward(tokens, mask, rm_cache)
        reward0 = (1 - alpha) * reward0.mean().item() + alpha * reward_threshold
        if debug: print(f'{reward0=:.2f}')

        def accept_check(reward, candidate):
            threshold = reward0 + (candidate.shape[1] - len_prompt) * (reward_threshold - reward0) / max_new_token
            threshold = min(threshold, reward_threshold)
            if debug: print(f'{reward=:.2f}, {threshold=:.2f}')

            if option == 'hard':
                return reward >= threshold
            elif option == 'soft':
                return random.uniform(0, 1) < min(1., exp((reward - threshold) / beta))
            else:
                assert False, 'Invalid reward sampling option!'

        while tokens.shape[1] - len_prompt < max_new_token:

            # sample a new candidate
            candidate = tokens.clone()
            candidate_mask = mask.clone()

            while candidate.shape[1] - tokens.shape[1] < 64:
                logits, llm_cache = self.from_token_to_logit(candidate, candidate_mask, llm_cache)
                num_llm_call += len(tokens)
                
                selected_token = self.from_logit_to_token(logits, top_k=topk, temperature=beta)
                candidate = torch.cat([candidate, selected_token], dim=-1)
                candidate_mask = torch.cat([candidate_mask, torch.ones_like(selected_token)], dim=-1)

                if debug: print(f'{selected_token.shape=}')

                if candidate.shape[1] - tokens.shape[1] >= 4 and '.' in self.from_token_to_text(selected_token):
                    del logits
                    break

            if debug: print(f'Segment length = {candidate.shape[1] - tokens.shape[1]}')

            # evaluate the candidate
            reward, rm_cache = self.from_token_to_reward(candidate, candidate_mask, rm_cache)
            num_rm_call += len(tokens)
            reward = reward.mean().item()

            if reward > best_reward:
                del best_candidate, best_candidate_mask
                best_candidate, best_candidate_mask, best_reward = candidate.clone(), candidate_mask.clone(), reward

            # accept/reject the candidate
            if num_regeneration >= 20:
                del tokens, mask, candidate, candidate_mask
                best_reward = -1e34
                tokens, mask = best_candidate, best_candidate_mask
                num_regeneration = 0
            elif accept_check(reward, candidate):
                del tokens, mask, best_candidate, best_candidate_mask
                best_candidate, best_candidate_mask, best_reward = None, None, -1e34
                tokens, mask = candidate, candidate_mask
                num_regeneration = 0
            else:
                del candidate, candidate_mask
                num_regeneration += 1
                if debug: print(f'Rejected {num_regeneration} times!')

        return self.from_token_to_text(tokens), (reward, num_llm_call, num_rm_call)

    # Item-level Best-of-N
    @torch.no_grad()
    def bon_generate(self, prompt, n:int=10, topk:int=20, beta:float=1.0, max_new_token:int=128):
        num_llm_call, num_rm_call = 0, 0
        llm_cache = None
        tokens_best, reward_best = None, -1e34

        for _ in range(n):
            tokens, mask = self.from_text_to_token(prompt)

            for _ in range(max_new_token):
                logits, llm_cache = self.from_token_to_logit(tokens, mask, llm_cache)
                num_llm_call += len(tokens)
                selected_token = self.from_logit_to_token(logits, top_k=topk, temperature=beta)

                tokens = torch.cat([tokens, selected_token], dim=-1)
                mask = torch.cat([mask, torch.ones_like(selected_token)], dim=-1)

            reward, _ = self.from_token_to_reward(tokens, mask)
            num_rm_call += len(tokens)
            reward = reward.mean().item()
            if reward > reward_best:
                tokens_best, reward_best = tokens.clone(), reward
            
        return self.from_token_to_text(tokens_best), (reward_best, num_llm_call, num_rm_call)

    # BOLT: Fast Energy-based Controlled Text Generation with Tunable Biases
    def bolt_generate(self, prompt, reward_threshold:float=8.5, topk:int=40, beta:float=0.7, max_new_token:int=128):
        tokens, mask = self.from_text_to_token(prompt)
        batch_size = tokens.shape[0]

        # gradient accumulation of the past candidates
        bias = torch.randn((batch_size, tokens.shape[1] + max_new_token, self.tokenizer.vocab_size), dtype=torch.bfloat16, requires_grad=True, device=tokens.device)
        optimizer = torch.optim.AdamW([bias], lr=0.1)

        best_candidate, best_candidate_mask, best_reward = None, None, -1e34
        num_operation = {
            "regeneration": 0,
            "llm_call": 0,
            "rm_call": 0,
            "rm_grad_cal": 0,
            "llm_grad_cal": 0,
        }
        llm_cache, rm_cache = None, None

        while True:

            # sample a new candidate
            candidate = tokens.clone()
            candidate_mask = mask.clone()

            while candidate.shape[1] - tokens.shape[1] < max_new_token:
                if candidate.shape[1] - tokens.shape[1] == max_new_token - 1:
                    full_logits, llm_cache = self.from_token_to_full_logit(candidate, candidate_mask, llm_cache)
                    logits = full_logits[:, -1]

                    # add unconditioned logits for the first token
                    logit_first_token = torch.zeros_like(logits).unsqueeze(-2).requires_grad_()
                    full_logits = torch.cat([logit_first_token, logit_first_token, full_logits[:, :-1, :]], dim=-2)
                else:
                    logits, llm_cache = self.from_token_to_logit(candidate, candidate_mask, llm_cache)

                num_operation["llm_call"] += batch_size

                # modify the logits based on gradient direction
                logits = logits + (1 - (candidate.shape[1] - tokens.shape[1]) / max_new_token) * bias[:, candidate.shape[1]]

                selected_token = self.from_logit_to_token(logits, top_k=topk, temperature=beta)
                candidate = torch.cat([candidate, selected_token], dim=-1)
                candidate_mask = torch.cat([candidate_mask, torch.ones_like(selected_token)], dim=-1)

            # STE soft onehot tokens
            soft_onehot_tokens = torch.nn.functional.one_hot(candidate, num_classes=self.tokenizer.vocab_size).to(torch.bfloat16).requires_grad_()            
            softmax_dist = F.softmax(full_logits / beta, dim=-1)
            soft_onehot_tokens = soft_onehot_tokens + softmax_dist - softmax_dist.detach()

            # evaluate the candidate
            embedding = self.from_onehot_token_to_embedding(soft_onehot_tokens, self.RM)
            reward, rm_cache = self.from_embedding_to_reward(embedding, candidate_mask, rm_cache)

            # calculate the gradient
            self.RM.zero_grad(set_to_none=True)
            soft_onehot_tokens.grad = None
            optimizer.zero_grad(set_to_none=True)
            bias.grad = torch.autograd.grad(reward.sum(), soft_onehot_tokens, retain_graph=False, create_graph=False)[0].to(torch.bfloat16)
            optimizer.step()

            num_operation["rm_call"] += batch_size
            reward = reward.mean().item()

            if reward > best_reward:
                del best_candidate, best_candidate_mask
                best_candidate, best_candidate_mask, best_reward = candidate.clone(), candidate_mask.clone(), reward

            # accept/reject the candidate
            if num_operation["regeneration"] >= 50:
                del tokens, mask, candidate, candidate_mask, bias
                tokens, mask = best_candidate, best_candidate_mask
                break
            elif random.uniform(0, 1) < min(1., exp((reward - reward_threshold) / beta)):
                del tokens, mask, best_candidate, best_candidate_mask, bias
                tokens, mask = candidate, candidate_mask
                best_reward = reward
                break
            else:
                del candidate, candidate_mask
                num_operation["regeneration"] += 1

        return self.from_token_to_text(tokens), best_reward, num_operation

    # Item-level gradient-guided rejection sampling
    def grad_rs_generate(
        self,
        prompt,
        reward_threshold: float = 8.5,
        lr: float = 1.0,
        beta: float = 0.7,
        topk: int = 40,
        max_new_token: int = 128,
    ):
        tokens, mask = self.from_text_to_token(prompt)
        batch_size = tokens.shape[0]

        # gradient direction of the last candidate
        grad, grad_count = None, 0
    
        best_candidate, best_reward = None, -1e34
        num_operation = {
            "regeneration": 0,
            "llm_call": 0,
            "rm_call": 0,
            "rm_grad_cal": 0,
            "llm_grad_cal": 0,
        }
        llm_cache, rm_cache = None, None

        while True:

            # sample a new candidate
            candidate = tokens.clone()

            while candidate.shape[1] - tokens.shape[1] < max_new_token:
                if candidate.shape[1] - tokens.shape[1] == max_new_token - 1:
                    full_logits, llm_cache = self.from_token_to_full_logit(candidate, mask, llm_cache)
                    logits = full_logits[:, -1]

                    # add unconditioned logits for the first token
                    logit_first_token = torch.zeros_like(logits).unsqueeze(-2).requires_grad_()
                    full_logits = torch.cat([logit_first_token, logit_first_token, full_logits[:, :-1, :]], dim=-2)
                else:
                    logits, llm_cache = self.from_token_to_logit(candidate, mask, llm_cache)

                num_operation["llm_call"] += batch_size

                # modify the logits based on gradient direction
                if grad is not None:
                    logits = logits + lr * (1 - (candidate.shape[1] - tokens.shape[1]) / max_new_token) / beta * grad[:, candidate.shape[1]]

                selected_token = self.from_logit_to_token(logits, top_k=topk, temperature=beta)
                candidate = torch.cat([candidate, selected_token], dim=-1)
                mask = torch.cat([mask, torch.ones_like(selected_token)], dim=-1)

            # STE soft onehot tokens
            soft_onehot_tokens = torch.nn.functional.one_hot(candidate, num_classes=self.tokenizer.vocab_size).to(torch.bfloat16).requires_grad_()            
            softmax_dist = F.softmax(full_logits / beta, dim=-1)
            soft_onehot_tokens = soft_onehot_tokens + softmax_dist - softmax_dist.detach()

            # evaluate the candidate
            embedding = self.from_onehot_token_to_embedding(soft_onehot_tokens, self.RM)
            reward, rm_cache = self.from_embedding_to_reward(embedding, mask, rm_cache)

            # calculate the gradient
            self.RM.zero_grad(set_to_none=True)
            soft_onehot_tokens.grad = None
            grad = torch.autograd.grad(reward.sum(), soft_onehot_tokens, retain_graph=False, create_graph=False)[0]
            # grad_count += 1
            # if grad is not None:
            #     grad = (1 - 1 / grad_count) * grad + (1 / grad_count) * new_grad
            # else:
            #     grad = new_grad

            num_operation["rm_call"] += batch_size
            reward = reward.mean().item()

            if reward > best_reward:
                del best_candidate
                best_candidate, best_reward = candidate.clone(), reward

            # accept/reject the candidate
            if num_operation["regeneration"] >= 50:
                del tokens, mask, candidate, grad
                tokens = best_candidate
                break
            elif random.uniform(0, 1) < min(1., exp((reward - reward_threshold) / beta)):
                del tokens, mask, best_candidate, grad
                tokens = candidate
                best_reward = reward
                break
            else:
                del candidate
                num_operation["regeneration"] += 1

        return self.from_token_to_text(tokens), best_reward, num_operation

    @torch.no_grad()
    def rm_score(self, text):
        tokens, mask = self.from_text_to_token(text)
        reward, _ = self.from_token_to_reward(tokens, mask)
        return reward
