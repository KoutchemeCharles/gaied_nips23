import time
import torch
from tqdm import tqdm
from src.repair.rl.reward import continual_reward
from src.repair.sft.prompting import (
    extract_incorrect_code,
    extract_problem_description, 
    extract_repaired_code
)

from warnings import warn 

def _training_loop(self, tokenizer, ppo_trainer, back_map):

    generation_kwargs = self._get_generation_kwargs("best")
    
    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        if epoch >= self.config.training.total_ppo_epochs:
            break

        query_tensors = batch["input_ids"]
        
        response_tensors = _get_responses(ppo_trainer, 
                                          query_tensors, 
                                          generation_kwargs)
            
        batch["response"] = tokenizer.batch_decode(response_tensors, 
                                                   skip_special_tokens=True)
        batch["query"] = tokenizer.batch_decode(query_tensors, 
                                                skip_special_tokens=True)

        qr = [q + r for q, r in zip(batch["query"], batch["response"])]

        repairs = extract_repaired_code(qr, force_end=True)
        incorrect_codes = extract_incorrect_code(qr)
        problem_statements = extract_problem_description(qr)

        rewards = []
        for ps, buggy, repair in zip(problem_statements, incorrect_codes, repairs):
            problem = back_map.get(ps, "")
            if not problem:
                warn("Could not assign reward")
                rewards.append(torch.tensor(0.0))
                continue
            
            """problem["generation"] = buggy
            uto_buggy = self.dataset_handler.check_correctness(problem, timeout=5)
            # baseline = continual_reward(buggy, buggy, uto_buggy, print_output=False)
            """
            problem["generation"] = repair           
            uto_repair = self.dataset_handler.check_correctness(problem, timeout=5)
            reward_repair = continual_reward(buggy, repair, uto_repair, print_output=True)
            
            rewards.append(reward_repair)

        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)
    
    ppo_trainer.save_pretrained(self.model_save_dir)



def _get_responses(ppo_trainer, query_tensors, generation_kwargs):
    # BUG in transformers trl: when providing a batch 
    # of examples, the number of returned sequences
    # is not taken into account when forwarding the
    # response back 

    if generation_kwargs["num_return_sequences"] == 1:
        response_tensors = ppo_trainer.generate(
            query_tensors,
            return_prompt=False,
            **generation_kwargs
        )
    else:
        raise ValueError("Not working yet...")
        response_tensors = []
        for query in query_tensors:
            response = ppo_trainer.generate(query, **generation_kwargs)
            response_tensors.append(response.squeeze()[len(query):])
        
    return response_tensors

def _get_generation_kwargs(self, type="best"):
    
    if type == "best":
        generation_kwargs = self.sft_exp.load_best_genconfig()
        del generation_kwargs.max_length
        generation_kwargs.min_length = -1
        generation_kwargs.num_return_sequences = 1
        generation_kwargs = generation_kwargs.to_dict()

    elif type == "default":
        # For the response generation we just use sampling and make sure top-k 
        # and nucleus sampling are turned off as well as a minimal length.

        generation_kwargs = {
            "min_length": -1,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": self.agent.tokenizer.eos_token_id,
            "max_new_tokens": 256,
        }

    return generation_kwargs


def duplicate(l, n):
    return [val for val in l for _ in range(n)]