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
            llm_dir                                                             ,
            cache_dir   = cache_dir                                             ,
            token       = access_token                                          ,
            torch_dtype = torch.bfloat16 if (fp_bit == 16) else torch.float32   ,
            load_in_8bit= (fp_bit == 8)                                         ,
            load_in_4bit= (fp_bit == 4)                                         ,
            device_map  = device_map
        )

        if rm_dir is not None:
            print('==> Loading RM...')
            self.RM = AutoModelForSequenceClassification.from_pretrained(
                rm_dir                                                              ,
                cache_dir   = cache_dir                                             ,
                token       = access_token                                          ,
                num_labels  = 1                                                     ,
                torch_dtype = torch.bfloat16 if (fp_bit == 16) else torch.float32   ,
                load_in_8bit= (fp_bit == 8)                                         ,
                load_in_4bit= (fp_bit == 4)                                         ,
                device_map  = device_map
            )

        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '</s>'})

            self.LLM.resize_token_embeddings(len(self.tokenizer))
            self.LLM.eval()

            if self.RM is not None:
                self.RM.resize_token_embeddings(len(self.tokenizer))
                self.RM.eval()

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
    def args_generate(self, prompt, args_weight:float=1.5, topk:int=40, max_new_token:int=128):
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

    # CARDS (ours)
    @torch.no_grad()
    def rs_generate(
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

    # Naive RS: Item-level Rejection Sampling
    @torch.no_grad()
    def naive_rs_generate(
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

    # Sentence-level Rejection Sampling
    @torch.no_grad()
    def sentence_rs_generate(
        self                                ,
        prompt                              ,
        option              : str   = 'soft',
        reward_threshold    : float = 8.0   ,
        alpha               : float = 1.0   ,
        beta                : float = 1.0   ,
        topk                : int   = 100   ,
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

    # Output Reward
    @torch.no_grad()
    def rm_score(self, text):
        tokens, mask = self.from_text_to_token(text)
        reward, _ = self.from_token_to_reward(tokens, mask)
        return reward

    # Output Uncertainty (incomplete)
    def show_uncertainty(self, text):
        pass

    # Update Tokens by the Gradients of Rewards (incomplete)
    def RM_update_tokens(self, text, num_iter:int=10, lr:float=0.1):

        # from sentences to tokens
        tokens = self.tokenizer(text, return_tensors='pt', padding=True).input_ids.to(self.RM.device)
        onehot_tokens = torch.nn.functional.one_hot(tokens, num_classes=self.tokenizer.vocab_size).to(torch.float32)
        onehot_tokens.requires_grad_()

        # update the hidden states
        batch_size = len(onehot_tokens)

        for _ in range(num_iter):
            self.RM.zero_grad()
            onehot_tokens.grad = None

            rewards = self.get_reward_from_onehot_tokens(onehot_tokens)

            loss = -torch.nn.functional.logsigmoid(rewards).sum()
            grad = torch.autograd.grad(loss, onehot_tokens, retain_graph=False, create_graph=False)[0]
            grad_length = torch.sqrt(torch.sum(grad * grad, dim=-1))
            grad_cos = torch.sum(grad * onehot_tokens, dim=-1) / grad_length
            A = torch.stack([tokens[0], grad_length[0], grad_cos[0]])
            print(A)
            # break

            onehot_tokens.data -= lr*grad
            onehot_tokens.data /= onehot_tokens.data.sum(dim=-1, keepdim=True)

            tokens = torch.argmax(onehot_tokens.data, dim=-1)
            print(self.tokenizer.batch_decode(tokens, skip_special_tokens=True), rewards)

        # return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
        text = []
        for i in range(tokens.shape[1]):
            text.append(self.get_text_from_tokens([tokens[0,i].item()])[0])
        return text

    # Update Embeddings by the Gradients of Rewards (incomplete)
    def RM_update_embeddings(self, text, num_iter:int=10, lr:float=0.1):

        # from sentences to embeddings
        tokens = self.tokenizer(text, return_tensors='pt', padding=True).input_ids.to(self.RM.device)
        embeddings = self.get_RM_embedding_from_tokens(tokens)

        # update the hidden states
        batch_size = len(embeddings)

        for _ in range(num_iter):
            self.RM.zero_grad()
            embeddings.grad = None

            rewards = self.get_reward_from_RM_embedding(embeddings)

            loss = -torch.nn.functional.logsigmoid(rewards).sum()
            grad = torch.autograd.grad(loss, embeddings, retain_graph=False, create_graph=False)[0]

            # grad_length = torch.sqrt(torch.sum(grad * grad, dim=-1))
            # embedding_length = torch.sqrt(torch.sum(embeddings * embeddings, dim=-1))

            # grad_cos = torch.sum(grad * embeddings, dim=-1) / grad_length / embedding_length

            grad_dot_product = torch.sum(grad * embeddings, dim=-1)

            # A = torch.stack([tokens[0], grad_length[0], grad_cos[0]])
            print(grad_dot_product)
            break

        # return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
        text = []
        for i in range(tokens.shape[1]):
            text.append(self.get_text_from_tokens([tokens[0,i].item()])[0])
        return text
