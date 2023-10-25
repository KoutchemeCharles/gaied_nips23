import re

# we use this as stop word since we
# prompt the model starting from "```python\n"
STOP_WORDS = ["```"]

def build_training_prompt(example):
    example = build_inference_example(example)
    prompt = example["query"]
    prompt += example['repair']
    prompt += "\n```"
    example["labels"] = prompt

    return example

def build_inference_example(example):
    s = example['source_code']
    prompt = "Repair the incorrect programs\n"
    prompt += example["prompt"]
    prompt += "\n**INCORRECT CODE:**\n" + f"```python\n{s}\n```"
    prompt += "\n**REPAIRED CODE:**\n"
    prompt += "```python\n"
    
    example["query"] = prompt 

    return example

def extract_problem_description(outputs):
    double_star = re.escape("**")
    start_beacon = "Repair the incorrect programs\n"
    end_beacon = f"\n{double_star}INCORRECT CODE:{double_star}\n"
    
    matches = [locate(start_beacon, end_beacon, output)
               for output in outputs]
    # return the full generation if we do not find anything 
    matches = [sorted(l, key=lambda x: len(x))[-1] 
               if len(l) else "" for l in matches]

    return matches 

def extract_incorrect_code(outputs):
    double_star = re.escape("**")
    start_beacon = f"\n{double_star}INCORRECT CODE:{double_star}\n```python\n"
    end_beacon = f"```"#\n{double_star}REPAIRED CODE:{double_star}\n```python\n"

    matches = [locate(start_beacon, end_beacon, output)
               for output in outputs]
    # return the full generation if we do not find anything 
    matches = [sorted(l, key=lambda x: len(x))[-1] 
               if len(l) else "" for l in matches]

    return matches 

def extract_repaired_code(outputs, force_end=False): 
    double_star = re.escape("**")
    start_beacon = f"\n{double_star}REPAIRED CODE:{double_star}\n```python\n"
    end_beacon = "\n```"

    if force_end:
        outputs = [o + end_beacon if not o.endswith(end_beacon) else o
                   for o in outputs]

    matches = [locate(start_beacon, end_beacon, output)
               for output in outputs]
    
    # return the full generation if we do not find anything 
    matches = [sorted(l, key=lambda x: len(x))[-1] 
               if len(l) else "" for l in matches]

    return matches 

def locate(start_beacon, end_beacon, string):
    regex = f"(?<={start_beacon})(.+?)(?={end_beacon})"
    matches = re.findall(regex, string, re.DOTALL)
    return [m for m in matches 
            if m and not (m.startswith(start_beacon) 
                          or m.endswith(end_beacon))]

