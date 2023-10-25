import gc 
import random
import torch

import numpy as np 

def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, and `torch`.

    Args:
        seed (`int`): The seed to set.
    """

    # Taken from
    # https://github.com/huggingface/trl/blob/b4899b29d246ff656ba736198a7730f9e96aa73f/trl/core.py#L233
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def claim_memory():
    gc.collect()
    torch.cuda.empty_cache()
