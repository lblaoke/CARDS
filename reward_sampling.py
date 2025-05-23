import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, BitsAndBytesConfig, OPTForCausalLM

from tqdm import tqdm
import random
from math import *
import gc

from base_class import BaseRewardSampling
from tools import uncertainty

class RewardSampling(BaseRewardSampling):
    def __init__(
            self,
            llm_dir: str,
            rm_dir: str = None,
            gold_rm_dir: str = None,
            dpo_dir: str = None,
            draft_dir: str = None,
            draft_r_dir: str = None,
            cache_dir: str = None,
            access_token: str = None,
            seed: int = 1,
        ):
        super(RewardSampling, self).__init__(seed=seed)

        self.model_kwargs = {
            'token': access_token,
            'cache_dir': cache_dir,
            'torch_dtype': torch.bfloat16,
            'device_map': 'auto',
            'attn_implementation': 'flash_attention_2',
            # 'quantization_config': BitsAndBytesConfig(load_in_8bit=True),
        }

        print('\n==> Loading Tokenizer...')
        self.tokenizer = AutoTokenizer.from_pretrained(llm_dir, token=access_token, cache_dir=cache_dir)

        print('\n==> Loading Base Model...')
        self.LLM = AutoModelForCausalLM.from_pretrained(llm_dir, **self.model_kwargs)

        if rm_dir is not None:
            print('\n==> Loading RM...')
            self.RM = AutoModelForSequenceClassification.from_pretrained(rm_dir, num_labels = 1, **self.model_kwargs)

        self.gold_rm_dir = gold_rm_dir

        if dpo_dir is not None:
            print('\n==> Loading DPO Checkpoint...')
            self.dpo_ckpt = AutoModelForCausalLM.from_pretrained(dpo_dir, **self.model_kwargs)

        if draft_dir is not None:
            print('\n==> Loading Draft Model...')
            self.draft = AutoModelForCausalLM.from_pretrained(draft_dir, **self.model_kwargs)
            self.draft_r = AutoModelForCausalLM.from_pretrained(draft_r_dir, **self.model_kwargs)

        # add padding token if not exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '</s>'})

            self.LLM.resize_token_embeddings(len(self.tokenizer))
            self.LLM.eval()

            if self.RM is not None:
                self.RM.resize_token_embeddings(len(self.tokenizer))
                self.RM.eval()

            if self.dpo_ckpt is not None:
                self.dpo_ckpt.resize_token_embeddings(len(self.tokenizer))
                self.dpo_ckpt.eval()
            
            if self.draft is not None:
                self.draft.resize_token_embeddings(len(self.tokenizer))
                self.draft.eval()

    def unload_all(self):
        del self.LLM, self.RM, self.dpo_ckpt, self.draft
        self.LLM, self.RM, self.dpo_ckpt, self.draft = None, None, None, None
        torch.cuda.empty_cache()
        gc.collect()

    def load_gold_rm(self):
        print('\n==> Loading Gold RM...')
        self.gold_rm = AutoModelForSequenceClassification.from_pretrained(self.gold_rm_dir, num_labels = 1, **self.model_kwargs)
        self.gold_rm.tokenizer = AutoTokenizer.from_pretrained(self.gold_rm_dir, **self.model_kwargs)

        self.gold_rm.tokenizer.add_special_tokens({'pad_token': '</s>'})
        self.gold_rm.resize_token_embeddings(len(self.gold_rm.tokenizer))
        self.gold_rm.eval()

    @torch.no_grad()
    def get_likelihood(self, text):
        tokens, mask = self.from_text_to_token(text)
        tokens, mask = torch.tensor(tokens, device=self.RM.device), torch.tensor(mask, device=self.RM.device)
        logits, _ = self.from_token_to_full_logit(tokens, mask)
        dist = F.softmax(logits, dim=-1)
        prob = dist[:, :-1, :].gather(-1, tokens[:, 1:].unsqueeze(-1)).squeeze(-1).detach()
        return prob.mean().item()

    @torch.no_grad()
    def get_reward(self, text):
        if self.RM.tokenizer is None:
            tokens, mask = self.from_text_to_token(text)
        else:
            out = self.RM.tokenizer(text, padding=True)
            tokens, mask = out.input_ids, out.attention_mask

        tokens, mask = torch.tensor(tokens, device=self.RM.device), torch.tensor(mask, device=self.RM.device)
        reward, _ = self.from_token_to_reward(tokens, mask)
        return reward.mean().item()
    
    @torch.no_grad()
    def get_gold_reward(self, text):
        if self.gold_rm.tokenizer is None:
            tokens, mask = self.from_text_to_token(text)
        else:
            out = self.gold_rm.tokenizer(text, padding=True)
            tokens, mask = out.input_ids, out.attention_mask

        tokens, mask = torch.tensor(tokens, device=self.gold_rm.device), torch.tensor(mask, device=self.gold_rm.device)
        reward, _ = self.from_token_to_gold_reward(tokens, mask)
        return reward.mean().item()

    # Vanilla LLM
    @torch.no_grad()
    def generate(self, prompt=None, tokens=None, mask=None, top_k:int=1, beta:float=0.8, max_new_token:int=128):
        if tokens is None or mask is None:
            tokens, mask = self.from_text_to_token(prompt)
        tokens, mask = torch.tensor(tokens, device=self.LLM.device), torch.tensor(mask, device=self.LLM.device)

        num_llm_call = 0
        llm_cache = None

        for _ in range(max_new_token):
            logits, llm_cache = self.from_token_to_logit(token=tokens, mask=mask, cache=llm_cache)
            num_llm_call += len(tokens)
            selected_token = self.from_logit_to_token(logits, top_k=top_k, temperature=beta)

            tokens = torch.cat([tokens, selected_token], dim=-1)
            mask = torch.cat([mask, torch.ones_like(selected_token)], dim=-1)

        return self.from_token_to_text(tokens), (num_llm_call, 0.)

    # Speculative Decoding
    @torch.no_grad()
    def sd_generate(
        self,
        prompt = None,
        tokens = None,
        mask = None,
        beta: float = 0.8,
        topk: int = 40,
        max_draft_token: int = 4,
        max_new_token: int = 128,
        bonus_token: bool = True,
    ):
        if tokens is None or mask is None:
            tokens, mask = self.from_text_to_token(prompt)
        tokens, mask = torch.tensor(tokens, device=self.LLM.device), torch.tensor(mask, device=self.LLM.device)
        batch_size, len_prompt = tokens.shape[0], tokens.shape[1]
        num_llm_call = 0
        llm_cache, draft_cache = None, None
        total_accepted_tokens = 0
        total_draft_tokens = 0
        while tokens.shape[1] - len_prompt < max_new_token:

            # sample a new draft candidate
            candidate = tokens.clone()
            candidate_mask = mask.clone()

            pre_len = candidate.shape[1]
            for _ in range(max_draft_token):
                logits, draft_cache = self.from_token_to_logit(candidate, candidate_mask, model=self.draft, cache=draft_cache)

                selected_token = self.from_logit_to_token(logits, top_k=topk, temperature=beta)
                candidate = torch.cat([candidate, selected_token], dim=-1)
                candidate_mask = torch.cat([candidate_mask, torch.ones_like(selected_token)], dim=-1)

            # verify the candidate
            logits, draft_cache = self.from_token_to_logit(
                token = candidate,
                mask = candidate_mask,
                model = self.draft,
                logits_to_keep = max_draft_token + 1,
                cache = draft_cache,
            )

            draft_log_dist = F.log_softmax(logits / beta, dim=-1)
            draft_log_prob = draft_log_dist[:, :-1, :].gather(-1, candidate[:, -max_draft_token:].unsqueeze(-1)).squeeze(-1).detach().cpu()

            logits, llm_cache = self.from_token_to_logit(
                token = candidate,
                mask = candidate_mask,
                logits_to_keep = max_draft_token + 1,
                cache = llm_cache,
            )
            num_llm_call += batch_size

            target_log_dist = F.log_softmax(logits / beta, dim=-1)
            target_log_prob = target_log_dist[:, :-1, :].gather(-1, candidate[:, -max_draft_token:].unsqueeze(-1)).squeeze(-1).detach().cpu()

            # accept/reject the candidate
            accept_num = 0
            for i in range(max_draft_token):
                if random.uniform(0, 1) < min(1., torch.exp(target_log_prob[:, i] - draft_log_prob[:, i])):
                    accept_num += 1
                else:
                    break
            tokens = candidate[:, :(pre_len + accept_num)]
            mask = candidate_mask[:, :(pre_len + accept_num)]

            total_accepted_tokens += accept_num
            total_draft_tokens += max_draft_token

            if bonus_token and (accept_num < max_draft_token):
                current_dist = torch.exp(target_log_dist[:, pre_len + accept_num - 1, :]) - torch.exp(draft_log_dist[:, pre_len + accept_num - 1, :])
                current_dist = torch.clamp(current_dist, min=0.0)
                current_dist = current_dist / torch.sum(current_dist, dim=-1)

                selected_token = self.from_logit_to_token(current_dist, temperature=beta, already_probs=True)
                tokens = torch.cat([tokens, selected_token], dim=-1)
                mask = torch.cat([mask, torch.ones_like(selected_token)], dim=-1)

        # print("Acceptance rate: %.3f" % (total_accepted_tokens / total_draft_tokens))

        return self.from_token_to_text(tokens), (num_llm_call, 0.)

    # Shifted Speculative Sampling
    @torch.no_grad()
    def sss_generate(
        self,
        prompt = None,
        tokens = None,
        mask = None,
        beta: float = 0.8,
        gamma: float = 1.0,
        topk: int = 40,
        max_draft_token: int = 4,
        max_new_token: int = 128,
        bonus_token: bool = True,
    ):
        if tokens is None or mask is None:
            tokens, mask = self.from_text_to_token(prompt)
        tokens, mask = torch.tensor(tokens, device=self.LLM.device), torch.tensor(mask, device=self.LLM.device)
        batch_size, len_prompt = tokens.shape[0], tokens.shape[1]
        num_llm_call = 0
        llm_cache, draft_cache, draft_r_cache = None, None, None
        total_accepted_tokens = 0
        total_draft_tokens = 0

        while tokens.shape[1] - len_prompt < max_new_token:
            candidate = tokens.clone()
            candidate_mask = mask.clone()
            pre_len = candidate.shape[1]

            # aligned draft model sample new draft tokens
            for _ in range(max_draft_token):
                logits, draft_r_cache = self.from_token_to_logit(candidate, candidate_mask, model=self.draft_r, cache=draft_r_cache)

                selected_token = self.from_logit_to_token(logits, top_k=topk, temperature=beta)
                candidate = torch.cat([candidate, selected_token], dim=-1)
                candidate_mask = torch.cat([candidate_mask, torch.ones_like(selected_token)], dim=-1)

            # aligned draft model verify the candidate
            logits, draft_r_cache = self.from_token_to_logit(
                token = candidate,
                mask = candidate_mask,
                model = self.draft_r,
                logits_to_keep = max_draft_token + 1,
                cache = draft_r_cache,
            )
            draft_r_log_dist = F.log_softmax(logits / beta, dim=-1)
            draft_r_log_prob = draft_r_log_dist[:, :-1, :].gather(-1, candidate[:, -max_draft_token:].unsqueeze(-1)).squeeze(-1).detach().cpu()

            # SFT draft model verify the candidate
            logits, draft_cache = self.from_token_to_logit(
                token = candidate,
                mask = candidate_mask,
                model = self.draft,
                logits_to_keep = max_draft_token + 1,
                cache = draft_cache,
            )
            draft_log_dist = F.log_softmax(logits / beta, dim=-1)
            draft_log_prob = draft_log_dist[:, :-1, :].gather(-1, candidate[:, -max_draft_token:].unsqueeze(-1)).squeeze(-1).detach().cpu()

            # Target model verify the candidate
            logits, llm_cache = self.from_token_to_logit(
                token = candidate,
                mask = candidate_mask,
                logits_to_keep = max_draft_token + 1,
                cache = llm_cache,
            )
            num_llm_call += batch_size
            target_log_dist = F.log_softmax(logits / beta, dim=-1)
            target_log_prob = target_log_dist[:, :-1, :].gather(-1, candidate[:, -max_draft_token:].unsqueeze(-1)).squeeze(-1).detach().cpu()

            # accept/reject the candidate
            accept_num = 0
            for i in range(max_draft_token):
                if random.uniform(0, 1) < min(1., torch.exp(target_log_prob[:, i] - draft_r_log_prob[:, i])):
                    accept_num += 1
                else:
                    break
            tokens = candidate[:, :(pre_len + accept_num)]
            mask = candidate_mask[:, :(pre_len + accept_num)]

            total_accepted_tokens += accept_num
            total_draft_tokens += max_draft_token

            if bonus_token and (accept_num < max_draft_token):
                current_dist = torch.exp(target_log_dist[:, pre_len + accept_num - 1, :] \
                                            + draft_r_log_dist[:, pre_len + accept_num - 1, :] \
                                            - draft_log_dist[:, pre_len + accept_num - 1, :] \
                                        ) \
                                # - torch.exp(draft_r_log_dist[:, pre_len + accept_num - 1, :])

                current_dist = torch.clamp(current_dist, min=0.0)
                current_dist = current_dist / torch.sum(current_dist, dim=-1)

                selected_token = self.from_logit_to_token(current_dist, temperature=beta, already_probs=True)
                tokens = torch.cat([tokens, selected_token], dim=-1)
                mask = torch.cat([mask, torch.ones_like(selected_token)], dim=-1)

        return self.from_token_to_text(tokens), (num_llm_call, 0.)

    # Item-level Best-of-N
    @torch.no_grad()
    def bon_generate(self, prompt, n:int=10, top_k:int=40, beta:float=1.0, max_new_token:int=128):
        num_llm_call, num_rm_call = 0, 0
        llm_cache = None
        tokens_best, reward_best = None, -1e34

        for _ in range(n):
            tokens, mask = self.from_text_to_token(prompt)
            tokens, mask = torch.tensor(tokens, device=self.LLM.device), torch.tensor(mask, device=self.LLM.device)

            for _ in range(max_new_token):
                logits, llm_cache = self.from_token_to_logit(tokens, mask, llm_cache)
                num_llm_call += len(tokens)
                selected_token = self.from_logit_to_token(logits, top_k=top_k, temperature=beta)

                tokens = torch.cat([tokens, selected_token], dim=-1)
                mask = torch.cat([mask, torch.ones_like(selected_token)], dim=-1)

            reward, _ = self.from_text_to_reward(self.from_token_to_text(tokens))
            num_rm_call += len(tokens)
            reward = reward.mean().item()
            if reward > reward_best:
                tokens_best, reward_best = tokens.clone(), reward

        print(f'{reward_best=}')
            
        return self.from_token_to_text(tokens_best), (num_llm_call, num_rm_call)

    # ARGS: Alignment as Reward-Guided Search
    @torch.no_grad()
    def token_bon_generate(self, prompt, args_weight:float=1.5, topk:int=40, max_new_token:int=128):
        tokens, mask = self.from_text_to_token(prompt)
        tokens, mask = torch.tensor(tokens, device=self.LLM.device), torch.tensor(mask, device=self.LLM.device)
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

        return self.from_token_to_text(tokens), (num_llm_call, num_rm_call)

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
        tokens, mask = torch.tensor(tokens, device=self.LLM.device), torch.tensor(mask, device=self.LLM.device)

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

        return self.from_token_to_text(tokens), (num_llm_call, num_rm_call)

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
        tokens, mask = torch.tensor(tokens, device=self.LLM.device), torch.tensor(mask, device=self.LLM.device)

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
            if num_regeneration >= 20:
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

        return self.from_token_to_text(tokens), (num_llm_call, num_rm_call)

    # Fixed-length segment rejection sampling
    @torch.no_grad()
    def seg_fix_rs_generate(
        self,
        prompt,
        method:str = 'rain',
        reward_threshold:float = 3.5,
        alpha:float = 0.5,
        beta:float = 0.7,
        topk:int = 40,
        max_new_token:int = 128,
    ):
        tokens, mask = self.from_text_to_token(prompt)
        tokens, mask = torch.tensor(tokens, device=self.LLM.device), torch.tensor(mask, device=self.LLM.device)

        len_prompt = tokens.shape[1]
        best_candidate, best_candidate_mask, best_reward = None, None, -1e34
        num_regeneration, num_llm_call, num_rm_call = 0, 0, 0
        llm_cache, rm_cache = None, None

        if method == 'treebon':
            reward0, rm_cache = self.from_token_to_weighted_implicit_reward(tokens, mask, rm_cache)
        elif method == 'rain':
            with open('f1.txt') as f:
                f1 = f.read()
            with open('f2.txt') as f:
                f2 = f.read()
            with open('r1.txt') as f:
                r1 = f.read()
            with open('r2.txt') as f:
                r2 = f.read()
            reward0, rm_cache = self.from_token_to_self_reward(f1, f2, r1, r2, self.from_token_to_text(tokens), rm_cache)
        else:
            assert False, 'Invalid method!'
        reward0 = (1 - alpha) * reward0.mean().item() + alpha * reward_threshold

        def accept_check(reward, candidate):
            threshold = reward0 + (candidate.shape[1] - len_prompt) * (reward_threshold - reward0) / max_new_token
            threshold = min(threshold, reward_threshold)

            return random.uniform(0, 1) < min(1., exp((reward - threshold) / beta))

        if method == 'treebon':
            seg_len = 32
        elif method == 'rain':
            seg_len = 10
        else:
            assert False, 'Invalid method!'

        while tokens.shape[1] - len_prompt < max_new_token:

            # sample a new candidate
            candidate = tokens.clone()
            candidate_mask = mask.clone()

            while True:
                logits, llm_cache = self.from_token_to_logit(candidate, candidate_mask, llm_cache)
                num_llm_call += len(tokens)
                
                selected_token = self.from_logit_to_token(logits, top_k=topk, temperature=beta)
                candidate = torch.cat([candidate, selected_token], dim=-1)
                candidate_mask = torch.cat([candidate_mask, torch.ones_like(selected_token)], dim=-1)

                if candidate.shape[1] - tokens.shape[1] == 32:
                    del logits
                    break

            # evaluate the candidate
            if method == 'treebon':
                reward, rm_cache = self.from_token_to_weighted_implicit_reward(candidate, candidate_mask, rm_cache)
            elif method == 'rain':
                reward, rm_cache = self.from_token_to_self_reward(f1, f2, r1, r2, self.from_token_to_text(candidate), rm_cache)
            else:
                assert False, 'Invalid method!'

            num_rm_call += 2 * len(tokens)
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

        return self.from_token_to_text(tokens), (num_llm_call, num_rm_call)

    # Sentence-level rejection sampling
    @torch.no_grad()
    def seg_punctuation_rs_generate(
        self                                ,
        prompt                              ,
        entropy_threshold   : float = 1.0   ,
        reward_threshold    : float = 8.5   ,
        alpha               : float = 0.5   ,
        beta                : float = 0.7   ,
        topk                : int   = 40    ,
        max_new_token       : int   = 128   ,
    ):
        tokens, mask = self.from_text_to_token(prompt)
        tokens, mask = torch.tensor(tokens, device=self.LLM.device), torch.tensor(mask, device=self.LLM.device)

        len_prompt = tokens.shape[1]
        best_candidate, best_candidate_mask, best_reward = None, None, -1e34
        num_regeneration, num_llm_call, num_rm_call = 0, 0, 0
        llm_cache, rm_cache = None, None

        reward0, rm_cache = self.from_token_to_reward(tokens, mask, rm_cache)
        reward0 = (1 - alpha) * reward0.mean().item() + alpha * reward_threshold
        # print(f'{reward0=:.2f}')

        def accept_check(reward, candidate):
            threshold = reward0 + (candidate.shape[1] - len_prompt) * (reward_threshold - reward0) / max_new_token
            threshold = min(threshold, reward_threshold)
            # print(f'{reward=:.2f}, {threshold=:.2f}')

            return random.uniform(0, 1) < min(1., exp((reward - threshold) / beta))

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

                # print(f'{selected_token.shape=}')

                if candidate.shape[1] - tokens.shape[1] < 4: continue
                if not any(c in {',', '.', ':', ';', '?', '!'} for c in self.from_token_to_text(selected_token)): continue
                # print(f'{self.from_token_to_text(selected_token)=}')
                # print(f'{uncertainty.entropy(logits).mean().item()=}')
                if uncertainty.entropy(logits).mean().item() < entropy_threshold: continue

                del logits
                break

            # print(f'Segment length = {candidate.shape[1] - tokens.shape[1]}')

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
                # print(f'Rejected {num_regeneration} times!')

        return self.from_token_to_text(tokens), (num_llm_call, num_rm_call)

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

        return self.from_token_to_text(tokens), num_operation

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

        return self.from_token_to_text(tokens), num_operation
