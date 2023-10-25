from random import sample
from src.utils.code import simple_clean

def create_few_shot_example(example, bank, k):
    fs_prompt = "Repair the incorrect programs\n"
    # the prompt already contains an indicator that it's the problem statement
    fs_prompt += example["prompt"] 
    if len(bank.get(example["problem_id"], [])) >= k:
        examples = sample(bank[example["problem_id"]], k)
        for inc, cor in examples:
            fs_prompt += "\n**INCORRECT CODE:**\n" + f"```python\n{simple_clean(inc)}\n```"
            fs_prompt += "\n**REPAIRED CODE:**\n" + f"```python\n{simple_clean(cor)}\n```"
            
    fs_prompt += "\n**INCORRECT CODE:**\n" + f"```python\n{simple_clean(example['source_code'])}\n```"
    fs_prompt += "\n**REPAIRED CODE:**\n"

    example["fs_prompt"] = fs_prompt.strip()
    return example



# TODO: context level prompting: add examples pairs using a tokenizer as long as it
# fits the context size 


def context_prompting(example, bank, tokenizer, context_size):
    """
    Add examples pairs using a tokenizer as long as it fits the context lenght
    """
    pass 