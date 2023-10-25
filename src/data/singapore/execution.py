from src.utils.code import does_compile
from src.utils.execution import exec_script

def grade(problem):
    if not does_compile(problem["generation"]):
        return 0.0 #"failed: code does not compile"

    tests = split_test_string(problem["test"])
    score = 0
    score = (sum([exec_script(f"{problem['generation']}\n{test}\n")[0]
                 for test in tests]) / len(tests)) * 100
    
    return score 
    
def split_test_string(test_string):
    inputs, ex_outputs, glob = get_inputs_and_expected_outputs(test_string)
    return [f"{glob}\nassert {inp}=={out}"for inp, out in zip(inputs, ex_outputs)]

def get_inputs_and_expected_outputs(test_string):
    glob = test_string[:test_string.find("assert")].strip()
    test_string = test_string[test_string.find("assert"):]
    test_string = test_string.replace("assert", "").strip()

    test_string = test_string.replace(") ==", ")==")
    test_string = test_string.split(" and ")
    
    inputs, outputs = zip(*[test_case.split(")==") 
                       for test_case in test_string])
    inputs = [i + ')' for i in inputs]

    return inputs, outputs, glob
