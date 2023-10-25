""" Utility function for evaluating repair strategy on a given dataset. """

from src.utils.evaluation import evaluate_functional_correctness

def _evaluate(self, generate, dataset):
    
    def get_repairs(example):
        """ 
        Complete the few_shot prompt to obtain the repairs.
        For each incorrect program, we generate multiple candidate repairs
        that need to be flattened for correctness evaluation.

        """
        
        inputs = self.agent.tokenizer(example["fs_prompt"], 
                                      return_tensors="pt")
        
        output_ids = generate(inputs)
        
        decoded_output = self.agent.decode(output_ids)
        generations = extract_repaired_code(decoded_output)
        new_output = {k: v * len(generations) for k, v in example.items()}
        new_output["generation"] = generations 
        return new_output
    
    eval_ds = dataset.map(get_repairs, batched=True, batch_size=2)

    pass_at_k = evaluate_functional_correctness(eval_ds.to_list(),
                                                1, 
                                                self.config.heval_k)
    eval_ds = eval_ds.remove_columns(["normalized"])
    results = {
        "experiment": "few_shot",
        "eval_ds": eval_ds.to_dict(),
        "pass_at_k": pass_at_k,
    }

    return results

def extract_code_blocks(text):
    code_blocks = text.split("```")
    code_blocks = [c.strip() for c in code_blocks]
    code_blocks = [c.replace("python", '') for c in code_blocks 
                   if c.startswith('python')]
    return code_blocks

def extract_repaired_code(outputs):
    beacon = "\n**REPAIRED CODE:**\n"
    new_outputs = []
    for output in outputs:
        response = output[output.rfind(beacon) + len(beacon):]
        code_blocks = extract_code_blocks(response)
        first_repair = code_blocks[0] if code_blocks else ""
        new_outputs.append(first_repair)

    return new_outputs