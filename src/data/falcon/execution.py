
import src.data.falcon.autograder as grader
from src.utils.files import write
from src.utils.code import does_compile
from src.utils.execution import exec_script

def grade(problem):
    if not does_compile(problem["generation"]):
        return 0.0 #"failed: code does not compile"

    write(problem["problem_id"] + ".py", problem["generation"])
    write("autograder.py", get_autograder_code())
    exec_string = create_execution_string(problem["testcase"])

    _, unit_test_output = exec_script(exec_string)
    score = get_unit_test_score(unit_test_output)

    # normalize the score
    score = score / problem["max_score"] * 100
    
    return score 
        
        
def create_execution_string(testcase):
    testcase = testcase.replace("from cs110 import autograder", "import autograder")
    testcase = testcase.replace("if __name__ == '__main__':", "")
    testcase = testcase.replace("result = test_passed()", "")
    testcase = testcase.replace('print("Unit Test Returned:", result)', "")
    testcase = testcase.strip()
    testcase = testcase + "\nresult = test_passed()\n"
    testcase = testcase + 'print("Unit Test Returned:", result)'
    
    return testcase

def get_autograder_code():
    """ Super dirty but temporary """
    with open(grader.__file__, "r") as fp:
        file_content = fp.read()
    return file_content

def get_unit_test_score(testcase_output):
    lines = testcase_output.splitlines()
    utr = [l for l in lines if l.startswith("Unit Test Returned:")]
    if utr:
        return float(utr[0].replace("Unit Test Returned:", "").strip())
    return 0.0