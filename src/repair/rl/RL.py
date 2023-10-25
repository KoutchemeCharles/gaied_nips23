""" 
Training procedure adapted from
https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama/scripts/rl_training.py
"""

from src.utils.files import save_json
from src.repair.Experiment import Experiment
from src.repair.sft.FinetunedRepair import FinetunedRepair
from src.repair.sft.prompting import STOP_WORDS

from accelerate import Accelerator
from trl import (
    PPOTrainer, PPOConfig, 
    AutoModelForCausalLMWithValueHead
)

class RL(Experiment):

    def __init__(self, config, test_run=False) -> None:
        super().__init__(config, "rl", test_run)
        self.sft_exp = FinetunedRepair(config, test_run=test_run)

    from .processing import build_train_dataset, get_test_dataset
    from .training import _training_loop, _get_generation_kwargs

    def train(self):        
        
        # 1. load a pretrained model, and its tokenizer
        model = self.load_model()
        tokenizer = self.agent.load_tokenizer()

        # 2. load the dataset
        dataset, back_map = self.build_train_dataset(tokenizer)
        
        # 3. define config
        ppo_config = self.load_ppo_config()
        ppo_trainer = PPOTrainer(
            ppo_config, model,
            ref_model=None, # an exact copy of the model will be created
            tokenizer=tokenizer,
            dataset=dataset,
            data_collator=collator,
        )
        
        self._training_loop(tokenizer, ppo_trainer, back_map)

    def evaluate(self):
        """ Evaluate the RL model on the test set. """
        
        model = self.load_trained_model().to(self.device) #.to(self.accelerator.device)
        test_dataset = self.get_test_dataset()
        gen_config = self.sft_exp.load_best_genconfig()
        generate = self.agent.get_generate(model, gen_config, STOP_WORDS)
        
        results = self.sft_exp._evaluate(generate, test_dataset)
        results["experiment"] = self.name 
        save_json(results, self.results_save_path)

    def load_ppo_config(self):
        ppo_config = self.config.training.ppo_config
        ppo_config.seed = self.config.seed 
        config = PPOConfig(**ppo_config)
        return config
    
    def load_model(self, current_device=None):
        """ 
        Load a model trained using SFT on the same dataset. 
        """

        if not current_device: current_device = Accelerator().process_index
        load_func = AutoModelForCausalLMWithValueHead.from_pretrained
        path = self.sft_exp.model_save_dir
        print("loading model from", path, "at", current_device)

        # Super important apparently to load the model with 
        # trust_remote_code = True otherwise generations are giberish?
        # Toke me way too long to figure that one out...
        return load_func(path, trust_remote_code=True)#, device_map={"": current_device})
    

def collator(data):
    """ Transform a list of dictionaries into a dictionary of lists. """
    return dict((key, [d[key] for d in data]) for key in data[0])
