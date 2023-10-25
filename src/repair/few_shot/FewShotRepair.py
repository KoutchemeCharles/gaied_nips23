""" Few-shot program repair """


from datasets import Dataset
from src.repair.Experiment import Experiment
from src.repair.few_shot.prompting import create_few_shot_example
from src.utils.files import save_json

class FewShotRepair(Experiment):

    def __init__(self, config, test=False) -> None:
        super().__init__(config, "few_shot", test)
        raise ValueError("Not updated with new repo structure")
        
    from .evaluate import _evaluate
    
    def train(self):
        """ Look for the best generation hyperparameters. """

        model = self.agent.load_model()
        train_df = self.dataset_handler.get_split("train")
        mapping = self.get_training_mapping(train_df)
        val_df = self.dataset_handler.get_split("val")  
        val_dataset = Dataset.from_pandas(val_df)
        val_dataset = val_dataset.filter(lambda ex: not ex["correct"])
        # Temporary while falconcode is being processed again 
        val_dataset = val_dataset.filter(lambda ex: ".csv" not in ex["prompt"] and "open" not in ex["source_code"])
        val_dataset = val_dataset.map(create_few_shot_example, 
                                      fn_kwargs={"bank": mapping, "k": 2})
        if self.test_run: val_dataset = val_dataset.select(range(10, 11))
        self.search_best_gen_params(model, val_dataset)

        
    def evaluate(self):
        model = self.agent.load_model()

        train_df = self.dataset_handler.get_split("train")
        test_df = self.dataset_handler.get_split("test")
        test_dataset = Dataset.from_pandas(test_df)
        test_dataset = test_dataset.filter(lambda ex: not ex["correct"])
        # Temporary while falconcode is being processed again 
        test_dataset = test_dataset.filter(lambda ex: ".csv" not in ex["prompt"] and "open" not in ex["source_code"])
        mapping = self.get_training_mapping(train_df)
        test_dataset = test_dataset.map(create_few_shot_example, 
                                        fn_kwargs={"bank": mapping, "k": 3})
        if self.test_run: test_dataset = test_dataset.select(range(10, 11))
        generation_config = self.load_best_genconfig()
        generate = self.agent.get_generate(model, generation_config)
        results = self._evaluate(generate, test_dataset)
        save_json(results, self.results_save_path)

