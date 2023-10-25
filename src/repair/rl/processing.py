from datasets import Dataset
from src.repair.sft.processing import clean_source_codes
from src.repair.sft.prompting import build_inference_example

def build_train_dataset(self, tokenizer):
    """
    Fetch and process the validation dataset
    of the base model trained using SFT. We use
    these as "new prompts" since they have not
    been used during training.
    """
    val_df = self.dataset_handler.get_split("val")
    val_dataset = Dataset.from_pandas(val_df)
    val_dataset = val_dataset.filter(lambda ex: not ex["correct"])
    val_dataset = val_dataset.map(clean_source_codes)
    val_dataset = val_dataset.map(build_inference_example)
    original_columns = val_dataset.column_names

    # we need to be able to know for each source code what's the problem,
    # test case, etc. But we'll loose that information when compiling
    # the dataloader for training, so we keep a mapping
    back_map = (val_dataset.to_pandas()
                .drop_duplicates("prompt")
                .set_index("prompt")
                .to_dict("index"))
    
    def preprocess_function(example):
        inputs = tokenizer(example["query"],
                            truncation=True)
        example["input_ids"] = inputs.input_ids
        example["input_query"] = tokenizer.decode(example["input_ids"])
        return example

    val_dataset = val_dataset.map(
        preprocess_function,
        batched=False,
        remove_columns=original_columns
    )

    val_dataset = val_dataset.shuffle(self.config.seed)
    val_dataset.set_format(type="torch")

    if self.test_run: val_dataset = val_dataset.select(range(10, 15))

    return val_dataset, back_map


def get_test_dataset(self):
    """
    Fetch and process the evaluation dataset
    """

    test_dataset = self.sft_exp.get_test_dataset()
    if self.test_run: test_dataset = test_dataset.select(range(10, 11))
    return test_dataset