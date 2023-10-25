import numpy as np
from src.repair.sft.prompting import extract_repaired_code
from src.utils.data_structures import find_closest_in_dict
from src.utils.execution import run_execution 
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import concatenate_datasets

def _get_trainer(self, dataset_dict, **kwargs):
    """ We finetune the pre-trained model for a few epochs
    on our dataset of solutions. """

    tokenizer = self.agent.load_tokenizer()
    tokenizer.padding_side='right' # to supress training warning and better for half precision training

    args = TrainingArguments(
        self.model_save_dir,
        overwrite_output_dir=True,
        evaluation_strategy = "epoch",
        save_strategy="no",
        seed=self.config.seed,
        report_to="wandb",
        **kwargs
    )

    # notes:
    # we do not pack if searching for best train param because we know
    # it will prevent us to properly estimate unit test outputs
    trainer = SFTTrainer(
        model=self.agent.hf_model_name, # Needed by SFTTRainer to not cause error 
        model_init=lambda trial: self.agent.load_model(),
        tokenizer=tokenizer,
        args=args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["test"],
        packing=True,#not self.config.training.wandb_hp_space, #
        max_seq_length=self.config.max_seq_length, 
        dataset_text_field="labels",
        #compute_metrics = self._get_compute_metrics(dataset_dict)
    )
    
    return trainer 



# map each incorrect program into the associated problem dictionary
# TODO: noticed it does not work because model generations are greedy
# and it would be the same as to maximize the original generations... 
# try another time 

def _get_compute_metrics(self, dataset_dict):
    """ New evaluation loss metric that returns the score obtained by
    the model generations through unit tests. Will allow us to search
    for which training parameters are the best. """

   

    # merge the two datasets
    dataset = concatenate_datasets(list(dataset_dict.values()))
    back_map = (dataset.to_pandas()
                .drop_duplicates("repair")
                .set_index("repair")
                .to_dict("index"))
    grader = self.dataset_handler.grade_fn

    def _compute_metrics(eval_pred):
        predictions, labels = eval_pred
        # first map the labels back to what the ideal repair should be 
        # predictions are the models logits returned
        # we need to map them back to token id labels 


        # https://github.com/huggingface/transformers/issues/24433
        
        # l in range(len(self.tokenizer))
        print("labels", labels)
        labels = [[l if l != -100 else self.agent.tokenizer.pad_token_id for l in ll]
                  for ll in labels]
        
        pred_labels = np.argmax(predictions, axis=-1)
        ideal_repairs = extract_repaired_code(self.agent.decode(labels))
        model_repairs = extract_repaired_code(self.agent.decode(pred_labels))

        for ideal, generation in zip(ideal_repairs, model_repairs):
            print("model greedy generations")
            print(generation)
            print("expected generations")
            print(ideal)
        
        scores = []
        for ideal, generation in zip(ideal_repairs, model_repairs):
            problem = find_closest_in_dict(back_map, ideal)
            problem["generation"], problem["completion_id"] = generation, 1
            uto = run_execution(grader, problem,  timeout=5)
            # score = 1.0 if uto["passed"] else 0.0

            # TODO: change this score now is available 
            print("unit test output", uto)
            if uto["passed"]:
                score = 1.0
            elif uto["result"].startswith("partial score"):
                score = float(uto["result"].replace("partial score", "")) / 100
            else:
                score = 0.0

            scores.append(score)

        return {"score": np.mean(scores)} # this will be summed by default_compute_objective
    
    return _compute_metrics


