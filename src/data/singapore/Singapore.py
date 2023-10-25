import os
from warnings import warn 
from datasets import (
    load_dataset, Dataset
)
from src.data.singapore.execution import grade
from src.utils.code import (
    does_compile, keep_unique_solutions
)
from src.utils.evaluation import evaluate_functional_correctness
class Singapore():

    splits = ["test"]

    def __init__(self, config) -> None:
        self.config = config 
        self.grade_fn = grade
        self.functional_form = True

        self.save_path = os.path.join(self.config.path, f"dataset.json")
        if self.config.reprocess or not os.path.exists(self.save_path):
            self.dataset = self._create_dataset()
            self.dataset.to_json(self.save_path)

        else:
            self.dataset = Dataset.from_json(self.save_path)

        
        
    def get_split(self, split):
        warn(f"Singapore doesn't have split {split}, returning singe test split")
        # if split != "test": raise ValueError(f"Only test split is available")
        return self.dataset.to_pandas()
        
    def _create_dataset(self):
        dataset = load_dataset("koutch/intro_prog", "singapore_data")
        dataset = dataset["train"] # the only available split
        dataset = self._rename_to_standard(dataset)
        dataset = dataset.filter(lambda ex: does_compile(ex["source_code"]))

        df = dataset.to_pandas()
        print("Original dataset statistics")
        print("Number of solutions", len(df["source_code"]))
        print("Number of problems", len(df["problem_id"]))
        print("Number of problems per assingment", df.groupby("problem_id").correct.value_counts())
        
        dataset = Dataset.from_pandas(keep_unique_solutions(dataset.to_pandas(), 
                                                            "source_code", "problem_id"))
        dataset = self._get_scores(dataset)
        dataset = dataset.filter(lambda ex: does_compile(ex["source_code"]))

        df = dataset.to_pandas()
        print("After removing duplicates")
        print("Number of solutions", len(df["source_code"]))
        print("Number of problems", len(df["problem_id"]))
        print("Number of problems per assingment", df.groupby("problem_id").correct.value_counts())


        return dataset 
    
    
    def _rename_to_standard(self, dataset):
        dataset = dataset.rename_column("description", "prompt")
        dataset = dataset.rename_column("func_code", "source_code")
        dataset = dataset.rename_column("assignment_id", "problem_id")
        dataset = dataset.rename_column("submission_id", "id")
        
        return dataset
    
    def _get_scores(self, dataset):
        dataset = dataset.rename_column("source_code", "generation")
        dataset = dataset.add_column("completion_id", [1] * len(dataset))
        _, dataset = evaluate_functional_correctness(dataset, self.grade_fn, k=[1])
        dataset = dataset.rename_column("generation", "source_code")
        dataset = dataset.rename_column("generation_score", "score")
        dataset = dataset.remove_columns(["completion_id", "generation_correct"])

        return dataset 
