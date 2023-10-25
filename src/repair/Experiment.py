""" Base class for running experiments """

import os
from torch.cuda import is_available
#from accelerate import Accelerator
from src.agent.Agent import Agent
from src.data.falcon.FalconCode import FalconCode
from src.data.mapping.Mapping import DefaultMapping
from src.data.mapping.refactory.Refactory import Refactory
from src.data.singapore.Singapore import Singapore
from src.utils.files import create_dir, save_json

class Experiment():
    
    def __init__(self, config, name, test_run=False) -> None:
        self.config = config
        self.test_run = test_run
        self.name = name + "_test_run" if self.test_run else name
        
        #self.accelerator = Accelerator()
        self.device = "cuda" if is_available() else "cpu"

        self.agent = self._load_agent_handler()
        self.dataset_handler = self._load_dataset_handler()
        self.mapping = self._load_mapping()
        
        
        self._init_directories()

    from .generation import (
        search_best_gen_params,
        _search_best_gen_params,
        load_best_genconfig,
    )

    from .training import (
        _train,
        _search_best_training_params,
        load_trained_model
    )

    def _load_agent_handler(self):
        return Agent(self.config.agent)#, self.accelerator)
    
    def _load_dataset_handler(self):
        ds_name = self.config.dataset.name
        if self.config.dataset.name.startswith("falcon"):
            return FalconCode(self.config.dataset)
        elif self.config.dataset.name.startswith("singapore"):
            return Singapore(self.config.dataset)
        else:
            raise ValueError(f"Unknown dataset {ds_name}")
    
    def _load_mapping(self, name=""):
        name = name if name else self.config.dataset.mapping.name
        if name.startswith("default"):
            return DefaultMapping(self.config.dataset.mapping, 
                                  self.dataset_handler)
        elif name.startswith("refactory"):
            return Refactory(self.config.dataset.mapping, 
                             self.dataset_handler)
        
    def _init_directories(self):
        """ Initialize all the saving directories. """
        
        self.save_dir = os.path.join(self.config.save_dir, self.name)
        self.results_save_dir = os.path.join(self.save_dir, "results")
        self.model_save_dir = os.path.join(self.save_dir, "model")
        self.cache_dir = os.path.join(self.save_dir, "cache")
        create_dir(self.save_dir)
        create_dir(self.cache_dir)
        create_dir(self.results_save_dir)
        create_dir(self.model_save_dir)
        filename = f"{self.config.dataset.name}_results.json"
        self.results_save_path = os.path.join(self.save_dir, "results", filename)

        # saving the configuration file in the directory
        save_json(self.config, os.path.join(self.save_dir, "experiment_config.json"))

