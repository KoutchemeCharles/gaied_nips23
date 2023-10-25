""" Evaluating code functional correctness

Taken and adapted from:
https://github.com/openai/human-eval/blob/master/human_eval/evaluation.py

"""
import tqdm
import resource
import itertools
import numpy as np
from collections import defaultdict
from typing import List, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils.core import claim_memory
from src.utils.execution import run_execution


def evaluate_functional_correctness(
    eval_ds, 
    grader,
    k: List[int] = [1, 10],
    timeout: float = 10.0):
    """
    Evaluates the functional correctness of generated samples.
    """
    #results = get_results(eval_ds, grader, timeout)
    results = get_results_with_thread(eval_ds, grader, timeout)
    pass_at_k = calculate_pass_at_k(results, k)
    eval_ds = add_exec_info_to_dataset(results, eval_ds)
    
    return pass_at_k, eval_ds


def calculate_pass_at_k(results, k):
    # Calculate pass@k.
    total, correct = [], []
    for _, result in results.items():
        passed = [r["passed"] for r in result]
        total.append(len(passed))
        correct.append(sum(passed))
    total = np.array(total)
    correct = np.array(correct)

    ks = k
    pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean()
                 for k in ks if (total >= k).all()}
    
    return pass_at_k

def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])

def add_exec_info_to_dataset(results, eval_ds):
    # create a mapping task_id, completion_id to whether that completion passed 
    scores_map = {}
    for task_id, result in results.items():
        m = {(task_id, r["completion_id"]): r["result"] for r in result}
        scores_map.update(m)

    # We map back the results of the completions to original dataframe
    # this will be useful for computing our metrics
    scores_map = dict(sorted(scores_map.items()))
    scores = list(scores_map.values())
    eval_ds = eval_ds.sort(["id", "completion_id"])
    eval_ds = eval_ds.add_column("generation_score", scores)
    eval_ds = eval_ds.add_column("generation_correct", 
                                 np.array(scores) == 100)
    
    return eval_ds


def chunkify(iterable, chunk):
    """ Chunks an iterable into pieces. """

    it = iter(iterable)
    while True:
        piece = list(itertools.islice(it, chunk))
        if piece:
            yield piece
        else:
            return


def get_results_with_thread(eval_ds, grader, timeout):
    """ Warning: this function uses so much memory it's insane... 
    skipping it for now and running normal executions
    
    https://bugs.python.org/issue37909
    """
    
    results = defaultdict(list)
    
    # Check the generated samples against test suites.
    with ThreadPoolExecutor() as executor:

        # We chunk the full iterable into pieces because
        # submitting a large list of task to the executors 
        # is ressource intensive (all future tasks are kept into
        # memory)
    
        for ds in chunkify(eval_ds.to_list(), 1000):

            futures = []

            print("Reading generations...")
            for sample in tqdm.tqdm(ds):
                args = (grader, sample, timeout)
                future = executor.submit(run_execution, *args)
                futures.append(future)

            print("Running test suites...")
            for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
                result = future.result()
                results[result["task_id"]].append(result)

            claim_memory() 
            print('Ressources being used', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024, 'MB')


    return results 


def get_results(eval_ds, grader, timeout):
    print("Passing generations through unit tests...")
    results = defaultdict(list)
    for sample in tqdm.tqdm(eval_ds.to_list()):
        res = run_execution(grader, sample, timeout)
        results[res["task_id"]].append(res)

    return results 