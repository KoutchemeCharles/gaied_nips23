""" Base Wrapper class around HuggingFace Decoder models functionalities """

import torch 
from sklearn.model_selection import ParameterGrid
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    StoppingCriteriaList
)
from src.utils.StoppingCriteria import StopWordsStoppingCriteria

class Agent():

    def __init__(self, config) -> None:
        self.config = config
        self.model = self.load_model()
        self.tokenizer = self.load_tokenizer()
        
    def encode(self, input):
        return self.tokenizer(input)
    
    def decode(self, output_ids):
        output_codes = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return list(output_codes)
    
    def load_model(self, path=None):
        path = path if path is not None else self.config.path
        return AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
    
    def load_tokenizer(self):
        # Padding to the left since we implement decoder models 
        tokenizer = AutoTokenizer.from_pretrained(self.config.path, 
                                                  padding_side='left')
        tokenizer.truncation_side = "left"
        if not tokenizer.pad_token: tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    
    def get_gen_param_space(self, num_return_sequences):
        """ Get the search space of generation hyperparameters. """

        # Exploring different kinds of generation
        # Note that beam_search decoding strategies work better
        # for neural translation models 

        default = self.get_default_gen(num_return_sequences)

        # multinomial sampling
        mns_grid = {
            "do_sample": [True],
            "top_p": [0.95, 0.98],
            "temperature": [0.4, 0.6, 0.8], 
            "max_new_tokens": [256],
            "num_return_sequences": [num_return_sequences],
            "eos_token_id": [self.tokenizer.eos_token_id],
            "pad_token_id": [self.tokenizer.eos_token_id],
        }

        param_grid = [mns_grid]
        
        return list(ParameterGrid(param_grid)), default
    

    def get_generate(self, model, gen_conf, stop_words=[]):
        """ 
        Return a generation function which completes the given input.

        Parameters
        ----------
        model: AutoModelForCausalLM
            Transformers Decoder only model
        generation_config: GenerationConfig
            Generation parameters to use for inference

        """
        
        device = model.device #if hasattr(model, "device") else self.device #self.accelerator.device
        print("Model is ran on", device)
        sw_ids = _encode_stop_words(self.tokenizer, stop_words)
        
        def generate(inputs):
            inputs = inputs.to(device)
            input_ids = inputs["input_ids"]
            stop_crit = _get_stopping_criteria_list(sw_ids, input_ids)

            with torch.no_grad():
                gen_outputs = model.generate(**inputs, 
                                             generation_config=gen_conf,
                                             stopping_criteria=stop_crit)
                return gen_outputs

        return generate
    
    def _postprocess(self, decoded_code):
        return decoded_code
    
    def get_default_gen(self, num_return_sequences):
        """See paper "Evaluating Large Language Models Trained on Code". """

        if num_return_sequences == 1:
            best_temp = 0.2
        elif num_return_sequences <= 5:
            best_temp = 0.4
        elif num_return_sequences <= 10:
            best_temp = 0.6
        elif num_return_sequences <= 100:
            best_temp = 0.8
        else:
            best_temp = 1.0

        default = {
            "do_sample": True,
            "top_k": 0.0,
            "top_p": 0.95,
            "temperature": best_temp,
            "max_new_tokens": 256,
            "num_return_sequences": num_return_sequences,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        return default
    
    @property
    def hf_model_name(self):
        return self.config.path 

def _encode_stop_words(tokenizer, stop_words):
    f = lambda w: tokenizer.encode(w, add_special_tokens=False)
    return list(map(f, stop_words))
            
def _get_stopping_criteria_list(sw_ids, input_ids):
    max_input_length = input_ids.size(1)
    stopping_criteria = None
    if sw_ids:
        stopping_criteria = StoppingCriteriaList()
        max_lengths = [max_input_length for l in range(len(input_ids))]
        ssc = StopWordsStoppingCriteria(max_lengths, sw_ids)
        stopping_criteria.append(ssc)

    return stopping_criteria
    
