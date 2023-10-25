from datasets import Dataset
from src.repair.sft.prompting import (
    build_inference_example, build_training_prompt
)
from src.utils.code import simple_clean

def get_training_dataset(self):
    """
    Fetch and process the training dataset for supervised
    fintuning. 
    
    We map each incorrect solution in the training part
    to it's closest incorrect solution in the same split.
    """
    train_df = self.dataset_handler.get_split("train")
    train_dataset = Dataset.from_pandas(train_df)
    train_dataset = train_dataset.filter(lambda ex: not ex["correct"])
    if self.test_run: train_dataset = train_dataset.select(range(10, 100))

    train_dataset = self.mapping.apply_mapping_to_dataset(train_dataset)

    train_dataset = train_dataset.filter(lambda ex: ex["repair"])
    train_dataset = train_dataset.map(clean_source_codes)
    train_dataset = train_dataset.map(build_training_prompt)
    train_dataset = train_dataset.shuffle(self.config.seed)
    return train_dataset


def get_val_dataset(self):
    """
    Fetch and process the validation dataset for supervised
    fintuning. 
    
    We map each incorrect solution in the validation part
    to it's closest incorrect solution in both the training
    and validation subsets.

    """
    val_df = self.dataset_handler.get_split("val")
    val_dataset = Dataset.from_pandas(val_df)
    val_dataset = val_dataset.filter(lambda ex: not ex["correct"])
    if self.test_run: val_dataset = val_dataset.select(range(10, 15))

    val_dataset = self.mapping.apply_mapping_to_dataset(val_dataset)
    val_dataset = val_dataset.filter(lambda ex: ex["repair"])
    val_dataset = val_dataset.map(clean_source_codes)
    val_dataset = val_dataset.map(build_training_prompt)
    
    # select a smaller subset otherwise it would take too much time
    # we choose examples which do not have too much failed
    f = lambda ex: ex["score"] >= ex["max_score"] // 4
    val_dataset = val_dataset.filter(f)
    val_dataset = val_dataset.shuffle(seed=self.config.seed)
    
    return val_dataset


def get_test_dataset(self):
    """
    Fetch and process the test dataset for final evaluation. 
    """

    # for all experiments until we have a final pipeline, to avoid leaking test info
    test_df = self.dataset_handler.get_split("test") 
    test_dataset = Dataset.from_pandas(test_df)
    test_dataset = test_dataset.filter(lambda ex: not ex["correct"])

    if self.test_run: test_dataset = test_dataset.select(range(10, 15))

    test_dataset = test_dataset.map(clean_source_codes)
    test_dataset = test_dataset.map(build_inference_example)
    test_dataset = test_dataset.shuffle(seed=self.config.seed)
    return test_dataset


def clean_source_codes(example):
    example["source_code"] = simple_clean(example["source_code"])
    if "repair" in example:
        example["repair"] = simple_clean(example["repair"])
    return example

def smart_sample(dataset, n):

    def take_highest_scoring(sub_df):
        return sub_df.sort_values(by="score").iloc[-1]
    
    df = dataset.to_pandas()
    # Take easiest cases to repair from the hardest exercises
    df = df.groupby("problem_id").apply(take_highest_scoring)
    df = df.sort_values(by="score", ascending=True).iloc[:n]
    return Dataset.from_pandas(df, preserve_index=False) 
