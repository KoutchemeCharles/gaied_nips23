""" Taken from
https://github.com/dpfried/incoder/blob/main/example_batched_usage.py
"""

import torch 
from typing import List
from transformers import StoppingCriteria

class StopWordsStoppingCriteria(StoppingCriteria):
    def __init__(self, init_lengths: List[int], stop_words_encoded: List[List[int]]):
        super().__init__()
        self.init_lengths = init_lengths
        if stop_words_encoded is None:
            stop_words_encoded = []
        else:
            assert isinstance(stop_words_encoded[0], list)
        assert isinstance(stop_words_encoded, list)
        self.stop_words_encoded = stop_words_encoded

    def _contains_stop_words(self, tokens: List[int]):
        if not bool(self.stop_words_encoded):
            return False
        for start_ix in range(len(tokens)):
            for swe in self.stop_words_encoded:
                if tokens[start_ix:start_ix+len(swe)] == swe:
                    return True
        return False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for init_length, i_tokens in zip(self.init_lengths, input_ids):
            if not self._contains_stop_words(i_tokens[init_length:].tolist()):
                return False
        return True