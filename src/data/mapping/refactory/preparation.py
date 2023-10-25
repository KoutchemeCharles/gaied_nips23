import os 
import io
import tokenize
from src.utils.files import create_dir, save 

def create_save_dir(dataset, save_path):
    """ Create a temporary directory where the Refactory tool is going to
    load the data and perform the repairs. 
    """

    create_dir(save_path, clear=True)
    dataframe = dataset.to_pandas()
    
    questions = sorted(dataframe["problem_id"].unique())
    # sort the questions by the ones which have at least one correct
    # code and one incorrect code
    questions = [q for q in questions 
                 if len(dataframe[(dataframe.problem_id == q) 
                                  & (~dataframe.correct)])]
        
    for i, question in enumerate(questions):
        question_path = os.path.join(save_path, f"question_{i + 1}")
        create_dir(question_path)
        create_ans_folder(dataframe, question, question_path)
        create_code_folder(dataframe, question, i + 1, question_path)
        create_description_file(dataframe, question_path, question)

        with open(os.path.join(question_path, 'assignment.txt'), 'w') as fp:
            fp.write(str(question))
    
    return questions

def create_description_file(dataframe, question_path, question):
    """ Creates a description.txt file which will contain the assignment/question/code description. """
    
    description = dataframe[dataframe.problem_id == question].prompt.iloc[0]
    save_path = os.path.join(question_path, 'description.txt')
    save(save_path, description)

def create_ans_folder(dataframe, question, question_path):
    """ Takes an assignment, and create the list of inputs and outputs. """

    ans_folder = os.path.join(question_path, "ans")
    create_dir(ans_folder)
    inputs, outputs = ["True"], ["True"]
    if inputs and outputs:
        get_file_path = lambda f: os.path.join(ans_folder, f)
        for i in range(len(inputs)):
            xxx = "{:03d}".format(i)
            save(get_file_path(f"input_{xxx}.txt"), inputs[i])
            save(get_file_path(f"output_{xxx}.txt"), outputs[i])

def create_code_folder(dataframe, question, question_id, question_path):
    """ Create the folder containing the code part. """

    code_folder = os.path.join(question_path, "code")
    create_dir(code_folder)
    
    # create the correct and wrong subdir
    for name, corr  in zip(["correct", "wrong"], [True, False]):
        corr_folder = os.path.join(code_folder, name)
        create_dir(corr_folder)

        # Get all (in)correct programs in my dataset
        df = dataframe[(dataframe.problem_id == question) 
                       & (dataframe.correct == corr)]
        
        # Save them 
        get_file_path = lambda f: os.path.join(corr_folder, f)
        for index, code in df[["id", "source_code"]].to_numpy():
            # the filename format will be the index of the dataframe 
            save(get_file_path(f"{name}_{question_id}_{index}.py"), code)
        
    # create the reference solution
    # for the reference we take the first correct program for simplicity 
    ref_folder = os.path.join(code_folder, "reference")
    create_dir(ref_folder)

    # as the reference solution take a random working solution 
    mask = ((dataframe.problem_id == question) 
             & (dataframe["correct"]))
    
    #ref = dataframe.loc[mask, "source_code"].iloc[-1]
    #save(os.path.join(ref_folder, "reference.py"), ref)
    
    # additionally, save the name of the original question into a file 
    save(os.path.join(code_folder, "metadata"), question)


def find_indentation(code):
    indentation = ""
    for token in tokenize.generate_tokens(io.StringIO(code).readline):
        if token.type == 5: #indent token
            indentation = token.string
            break
    return indentation


def put_in_function(example):
    s = example["source_code"]

    p = "def wrapper(b):\n"
    indentation = find_indentation(s)
    if not indentation: indentation = "  "
    sc = "\n".join([indentation + line for line in s.splitlines()])
    s = p + sc + f"\n{indentation}return True\n"

    example["source_code"] = s
    return example


def extract_from_function(source_code):
    indentation = find_indentation(source_code)
    source_code = source_code.replace("def wrapper(b):\n", "")
    source_code = source_code.replace(f"\n{indentation}return True\n", "")
    source_code = "\n".join([s.replace(indentation, "", 1) for s in source_code.splitlines() if s])

    return source_code
