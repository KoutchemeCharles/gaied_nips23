from datasets import DatasetDict
from src.repair.Experiment import Experiment
from src.repair.sft.processing import smart_sample
from src.repair.sft.prompting import STOP_WORDS
from src.utils.files import save_json

class FinetunedRepair(Experiment):
    """ Supervised learning on a Introduction dataset. """
    
    def __init__(self, config, test_run=False) -> None:
        super().__init__(config, "sft", test_run)

    from .processing import (
        get_training_dataset,
        get_val_dataset,
        get_test_dataset
    )
    from .training import _get_trainer, _get_compute_metrics
    from .evaluate import _evaluate

    def train(self):
        """ 
        Train a base autoregressive model on pairs
        of incorrect programs and their associated repairs. 
        """
        
        train_dataset = self.get_training_dataset()
        val_dataset = self.get_val_dataset()

        dataset_dict = DatasetDict({
            "train": train_dataset,
            "test": val_dataset
        })

        model = self._train(dataset_dict)

        # we sample a small number becaus evaluation is expensive 
        val_dataset = smart_sample(val_dataset, 100)
        self.search_best_gen_params(model, val_dataset)


    def evaluate(self):
        """ Evaluate the model on the test set. """
        
        model = self.load_trained_model()
        test_dataset = self.get_test_dataset()
        
        generation_config = self.load_best_genconfig()
        generate = self.agent.get_generate(model, generation_config, STOP_WORDS)
        results = self._evaluate(generate, test_dataset)
        save_json(results, self.results_save_path)

