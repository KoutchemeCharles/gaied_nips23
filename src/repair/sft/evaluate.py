""" Utility function for evaluating repair strategy on a given dataset. """

from src.utils.core import claim_memory
from src.repair.sft.prompting import extract_repaired_code
from src.utils.evaluation import evaluate_functional_correctness

def _evaluate(self, generate, dataset):
    
    def get_repairs(example):
        """ 
        Complete the prompt to obtain the repairs 

        """
        
        inputs = self.agent.tokenizer(example["query"], 
                                      return_tensors="pt",
                                      truncation=True,
                                      padding=True,
                                      pad_to_multiple_of=8, # TODO: change that dynamically depending on numerical type
                                      max_length=self.config.max_seq_length)
        outputs = generate(inputs)
        # Reminder: model generations are always flattened into a 2d array
        # batch of 5 x (10 generations) = 50 arrays of size padding size
        # if we want to broadcast the generations back to their associated
        # problems we need to duplicate n_generatiosn times
        # each value in each value array of the examples dictionary 
        n_generations = len(outputs) // len(example["query"])

        decoded_outputs = self.agent.decode(outputs)
        generations = extract_repaired_code(decoded_outputs)
        new_output = {k: duplicate(v, n_generations) 
                      for k, v in example.items()}

        new_output["generation"] = generations
        # add a completion id to track the generations 
        cids = list(range(n_generations)) * len(example["query"])
        new_output["completion_id"] = cids

        return new_output

    batch_size = 4
    eval_ds = dataset.map(get_repairs, batched=True, 
                          batch_size=batch_size)
    
    grade_fn = self.dataset_handler.grade_fn
    heval_k = self.config.heval_k
    pass_at_k, eval_ds = (
        evaluate_functional_correctness(eval_ds, grade_fn, heval_k))
    
    claim_memory()

    # The following column cannot be serialized by json (bytes)
    if "normalized" in eval_ds.features:
        eval_ds = eval_ds.remove_columns(["normalized"])

    results = {
        "experiment": self.name,
        "eval_ds": eval_ds.to_dict(),
        "pass_at_k": pass_at_k,
    }

    
    return results


def duplicate(l, n):
    return [val for val in l for _ in range(n)]
