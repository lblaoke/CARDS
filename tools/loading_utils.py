import torch
import numpy as np
import random

def _gpu_init(seed:int=None):
    assert torch.cuda.is_available(), 'GPU not available!'
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    if seed is not None:
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
