import torch
from src.utils.code import does_compile
from src.utils.distance import rougelcsum_dist

def continual_reward(buggy, generation, uto, print_output=False):
    """ Reward the generation based on the unit test outputs results. """
   
    # TODO: score now contains the score 
    reward = 0.0
    if not generation or not does_compile(generation) or uto["result"].startswith("failed:"): 
        reward = -1.0
    elif uto["result"].startswith("partial score"):
        reward = float(uto["result"].replace("partial score", "")) / 100
    else:
        print("best kind of reward")
        reward = 1.0 + (1 - rougelcsum_dist(buggy, generation))

    if print_output:
        print()
        print()
        print("buggy")
        print(buggy)
        print("generation")
        print(generation)
        print("unit test output")
        print(uto)
        print("reward")
        print(reward)
        print("-------")
    
    return torch.tensor(reward)


def soft_continual_reward(buggy, generation, uto, factor=0.5, print_output=False):
    correctness_reward = 0.0
    if not generation or not does_compile(generation) or uto["result"].startswith("failed:"): 
        correctness_reward = -1.0
    elif uto["result"].startswith("partial score"):
        correctness_reward = float(uto["result"].replace("partial score", "")) / 100
    else:
        correctness_reward = 1.0
        print("best kind of reward")
        reward = 1.0 + (1 - rougelcsum_dist(buggy, generation))

    if print_output:
        print()
        print()
        print("buggy")
        print(buggy)
        print("generation")
        print(generation)
        print("unit test output")
        print(uto)
        print("reward")
        print(reward)
        print("-------")
    
    return torch.tensor(reward)


def coderl_reward(buggy, generation, uto):
    reward = 0.0
    if uto["result"].startswith("failed:"): 
        reward = -1.0
    elif uto["result"].startswith("partial score"):
        reward = -0.5
    else:
        reward = 1.0 - rougelcsum_dist(buggy, generation)

    return torch.tensor(reward)