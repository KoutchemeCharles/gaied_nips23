""" Utilities for model generation """

import numpy as np
from transformers import GenerationConfig


def search_best_gen_params(self, model, dataset):
    """ 
    Searches for the generation parameters which will
    produce the best results and saves them. 
    """

    nrs = max(self.config.heval_k) if self.config.heval_k else 1
    search_space, gen_params = self.agent.get_gen_param_space(nrs)

    if self.config.training.search_best_genconfig:
        device = self.device #self.accelerator.device
        model = model.to(device)
        gen_params = self._search_best_gen_params(model, dataset, search_space)
        print("Best generation parameters: ", gen_params)
    
    # save the best generation hyperparameters
    # at the same position than the model 
    generation_config = GenerationConfig(**gen_params)
    generation_config.save_pretrained(self.model_save_dir)
    print("Saved generation parameters at", self.model_save_dir)
    
    return generation_config


def _search_best_gen_params(self, model, dataset, gen_param_space):
    """ 
    Searches for the generation parameters which will
    produce the best results.
    """

    # When searching for the best generation hyperparameters
    # we can use the pass rate as a measure of success since at this
    # stage (end of training) we expect the model to be somewhat good
    # at generating repairs
    
    scores = []
    for i, gen_params in enumerate(gen_param_space):
        generation_config = GenerationConfig(**gen_params)
        generate = self.agent.get_generate(model, generation_config)
        print("evaluating config", generation_config)
        res = self._evaluate(generate, dataset)["pass_at_k"]
        scores.append(res[f"pass@{gen_params['num_return_sequences']}"])
        if self.test_run and i == 1:
            break

    return gen_param_space[np.argmax(scores)]

def load_best_genconfig(self):
    gen_config = GenerationConfig.from_pretrained(self.model_save_dir)
    print("Loaded generation config", gen_config)
    return gen_config