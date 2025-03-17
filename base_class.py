import torch
import torch.nn.functional as F
import transformers
from tools import loading_utils

class BaseRewardSampling:
    def __init__(self, seed):
        loading_utils._gpu_init(seed)

        self.device = None
        self.use_cache = False # KV cache is disabled by default, due to the repeating issue in the output

        self.tokenizer = None
        self.LLM = None
        self.RM = None
        self.dpo_ckpt = None
        self.draft = None

    ###################
    # basic functions #
    ###################

    def from_text_to_token(self, text):
        out = self.tokenizer(text, return_tensors='pt', padding=True)
        tokens, mask = out.input_ids.to(self.device), out.attention_mask.to(self.device)
        return tokens, mask

    def from_token_to_text(self, token):
        return self.tokenizer.batch_decode(token, skip_special_tokens=True)

    def from_token_to_embedding(self, token, model):
        return model.model.embed_tokens(token)

    def from_onehot_token_to_embedding(self, onehot_token, model):
        return onehot_token @ model.model.embed_tokens.weight

    # https://stackoverflow.com/questions/64523788/how-to-invert-a-pytorch-embedding
    def from_embedding_to_token(self, embedding, model):
        weights = model.model.embed_tokens.weight.data

        assert len(weights.shape) == 2, 'Invalid weight dimensions!'
        assert len(embedding.shape) == 3, 'Invalid embedding dimensions!'

        embedding = embedding.reshape(*embedding.shape[:2], 1, embedding.shape[-1])
        return torch.argmin(torch.norm(weights - embedding, dim=-1), dim=-1)

    def from_embedding_to_logit(self, embedding, mask=None, cache=None):
        out = self.LLM(
            input_ids = None,
            attention_mask = mask,
            past_key_values = cache,
            inputs_embeds = embedding,
            use_cache = self.use_cache,
            num_logits_to_keep = 1,
        )
        logits, cache = out.logits[:, -1], out.past_key_values
        del out
        return logits, cache

    def from_embedding_to_reward(self, embedding, mask=None, cache=None):
        out = self.RM(
            input_ids = None,
            attention_mask = mask,
            past_key_values = cache,
            inputs_embeds = embedding,
            use_cache = self.use_cache,
        )
        reward, cache = out.logits.flatten(), out.past_key_values
        del out
        return reward, cache

    def from_token_to_logit(self, token, mask=None, model=None, logits_to_keep:int=1, cache=None):
        kwargs = {
            'input_ids': token,
            'attention_mask': mask,
            'past_key_values': cache,
            'use_cache': self.use_cache,
            'num_logits_to_keep': logits_to_keep,
        }
        if logits_to_keep < 0 or isinstance(self.LLM, transformers.OPTForCausalLM):
            kwargs.pop('num_logits_to_keep', None)

        out = self.LLM(**kwargs) if model is None else model(**kwargs)
        logits, cache = out.logits, out.past_key_values
        del out

        if logits_to_keep == 1:
            logits = logits[:, -1]

        return logits, cache

    def from_token_to_reward(self, token, mask=None, cache=None):
        out = self.RM(
            input_ids = token,
            attention_mask = mask,
            past_key_values = cache,
            use_cache = self.use_cache,
        )
        reward, cache = out.logits.flatten(), out.past_key_values
        del out
        return reward, cache

    def from_logit_to_token(
        self,
        logit,
        discard_token = None, # set a specific token as -inf to discard it
        top_k:int = 1,
        temperature:float = 1.,
        return_prob:bool = False, # also return the probability of the selected token
    ):
        if discard_token is not None:
            batch_idx = torch.arange(logit.shape[0], dtype=discard_token.dtype, device=discard_token.device)
            discard_idx = torch.stack([batch_idx, discard_token.flatten()]).T
            logit[discard_idx[:, 0], discard_idx[:, 1]] = logit.min()

        if top_k == 1:
            selected_token = torch.argmax(logit, dim=-1, keepdim=True)
        else:
            val, idx = torch.topk(logit, k=top_k, dim=-1)
            selected_idx = torch.multinomial(F.softmax(val / temperature, dim=-1), num_samples=1)
            selected_token = torch.gather(idx, -1, selected_idx)

        if return_prob:
            prob = F.softmax(logit / temperature, dim=-1)
            return selected_token, prob.gather(-1, selected_token)
        else:
            return selected_token

    ######################
    # advanced functions #
    ######################

    def from_token_to_full_logit(self, token, mask=None, logits_to_keep:int=-1, cache=None):
        return self.from_token_to_logit(token, mask, self.LLM, logits_to_keep, cache)

    # RAIN (https://arxiv.org/abs/2309.07124)
    def from_token_to_self_reward(self, f1, f2, r1, r2, text, cache=None):
        reward = []
        for t in text:
            text1 = f1 + '\n\n' + t + '\n' + r1
            text2 = f2 + '\n\n' + t + '\n' + r2

            token1, mask1 = self.from_text_to_token(text1)
            token2, mask2 = self.from_text_to_token(text2)
            id_A = self.from_text_to_token('A')[0][0, -1].item()
            id_B = self.from_text_to_token('B')[0][0, -1].item()

            out1 = self.LLM(
                input_ids = token1,
                attention_mask = mask1,
                past_key_values = cache,
                use_cache = self.use_cache,
                num_logits_to_keep = 1,
            )
            out2 = self.LLM(
                input_ids = token2,
                attention_mask = mask2,
                past_key_values = cache,
                use_cache = self.use_cache,
                num_logits_to_keep = 1,
            )

            log_prob1_A = out1.logits[0, -1, id_A].reshape(1)
            log_prob1_B = out1.logits[0, -1, id_B].reshape(1)
            log_prob2_A = out2.logits[0, -1, id_A].reshape(1)
            log_prob2_B = out2.logits[0, -1, id_B].reshape(1)

            r_A = log_prob1_A - log_prob1_B
            r_B = log_prob2_B - log_prob2_A

            reward.append((r_A + r_B) / 2)

        return torch.cat(reward), out1.past_key_values

    # TreeBoN (https://arxiv.org/abs/2410.16033)
    def from_token_to_weighted_implicit_reward(self, token, mask=None, cache=None, prompt_len:int=0):
        out = self.LLM(
            input_ids = token,
            attention_mask = mask,
            past_key_values = cache,
            use_cache = self.use_cache,
        )
        dpo_out = self.dpo_ckpt(
            input_ids = token,
            attention_mask = mask,
        )

        log_prob = F.log_softmax(out.logits, dim=-1)
        dpo_log_prob = F.log_softmax(dpo_out.logits, dim=-1)

        log_prob = log_prob.gather(-1, token.unsqueeze(-1)).squeeze(-1)[:, prompt_len:]
        dpo_log_prob = dpo_log_prob.gather(-1, token.unsqueeze(-1)).squeeze(-1)[:, prompt_len:]

        w = torch.tensor([[i for i in range(1, log_prob.shape[-1] + 1)]], device=log_prob.device, dtype=log_prob.dtype)
        w = torch.reciprocal(w)

        reward = torch.sum(w * (dpo_log_prob - log_prob))

        return reward, out.past_key_values
