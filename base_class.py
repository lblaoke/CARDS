import torch
import torch.nn.functional as F
import numpy as np
import random

def GPU_setup(seed:int=None):
    assert torch.cuda.is_available(), 'GPU not available!'
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    if seed:
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        print(f'==> Random seed: {seed}')

class BaseRewardSampling:
    def __init__(self, seed):
        GPU_setup(seed)
        self.tokenizer  = None
        self.LLM        = None
        self.RM         = None

    #####################
    # basic functions
    #####################

    def from_text_to_token(self, text):
        out = self.tokenizer(text, return_tensors='pt', padding=True)
        tokens, mask = out.input_ids.to(self.LLM.device), out.attention_mask.to(self.LLM.device)
        return tokens, mask

    def from_token_to_text(self, token):
        return self.tokenizer.batch_decode(token, skip_special_tokens=True)

    def from_token_to_onehot_token(self, token):
        onehot_tokens = F.one_hot(token, num_classes=self.tokenizer.vocab_size)

    def from_onehot_token_to_token(self, onehot_token):
        return torch.argmax(onehot_token.data, dim=-1)

    def from_token_to_embedding(self, token, model):
        return model.model.embed_tokens(token)

    def from_onehot_token_to_embedding(self, onehot_token, model):
        assert onehot_token.dtype == torch.float32, 'onehot_tokens must be of type torch.float32, not %s!' % onehot_token.dtype

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
            input_ids           = None                  ,
            attention_mask      = mask                  ,
            past_key_values     = cache                 ,
            inputs_embeds       = embedding             ,
            use_cache           = (cache is not None)   ,
            num_logits_to_keep  = 1
        )
        logits, cache = out.logits[:, -1], out.past_key_values
        del out
        return logits, cache

    def from_embedding_to_reward(self, embedding, mask=None, cache=None):
        prepared_inputs = self.LLM.prepare_inputs_for_generation(input_ids=None, inputs_embeds=embedding, attention_mask=mask, past_key_values=cache, use_cache=(cache is not None))
        
        if 'cache_position' in prepared_inputs:
            del prepared_inputs['cache_position']
        
        out = self.RM(**prepared_inputs)
        reward, cache = out.logits.flatten(), out.past_key_values
        del out
        return reward, cache

    def from_token_to_logit(self, token, mask=None, cache=None):
        out = self.LLM(
            input_ids           = token                 ,
            attention_mask      = mask                  ,
            past_key_values     = cache                 ,
            use_cache           = (cache is not None)   ,
            num_logits_to_keep  = 1
        )
        logits, cache = out.logits[:, -1], out.past_key_values
        del out
        return logits, cache

    def from_token_to_reward(self, token, mask=None, cache=None):
        prepared_inputs = self.LLM.prepare_inputs_for_generation(input_ids=token, attention_mask=mask, past_key_values=cache, use_cache=(cache is not None))
        
        if 'cache_position' in prepared_inputs:
            del prepared_inputs['cache_position']

        out = self.RM(**prepared_inputs)
        reward, cache = out.logits.flatten(), out.past_key_values
        del out
        return reward, cache

    def from_logit_to_token(self, logit, discard_token=None, top_k:int=1, temperature:float=1.):
        if discard_token is not None:
            batch_idx = torch.arange(logit.shape[0], dtype=discard_token.dtype, device=discard_token.device)
            discard_idx = torch.stack([batch_idx, discard_token.flatten()]).T
            logit[discard_idx[:, 0], discard_idx[:, 1]] = logit.min()

        if top_k == 1:
            return torch.argmax(logit, dim=-1, keepdim=True)
        else:
            val, idx = torch.topk(logit, k=top_k, dim=-1)
            selected_idx = torch.multinomial(F.softmax(val / temperature, dim=-1), num_samples=1)
            selected_token = torch.gather(idx, -1, selected_idx)

            return selected_token

    #####################
    # advanced functions
    #####################

    def from_embedding_to_full_logit(self, embedding, mask=None):
        prepared_inputs = self.LLM.prepare_inputs_for_generation(input_ids=None, inputs_embeds=embedding, attention_mask=mask, use_cache=False)
        return self.LLM(**prepared_inputs).logits

    def from_token_to_full_logit(self, token, mask=None):
        prepared_inputs = self.LLM.prepare_inputs_for_generation(input_ids=token, attention_mask=mask, use_cache=False)
        return self.LLM(**prepared_inputs).logits
