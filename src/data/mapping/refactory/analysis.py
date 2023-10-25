import os
import pandas as pd 
from warnings import warn

from datasets import Dataset
from src.utils.code import simple_clean
from src.utils.evaluation import evaluate_functional_correctness


def collate_individual_csvs(dir_path):
    questions = os.listdir(dir_path)
    questions = [q for q in questions if q.startswith("question")]
    key_f = lambda q: int(q.split('_')[-1])
    questions = sorted(questions, key=key_f)

    dataframe = []
    for q in questions:
        q_path = os.path.join(dir_path, q, 'refactory_offline.csv')
        if not os.path.exists(q_path):
            warn(f"Results for assignment {q} are not available")
            continue
        dataframe.append(pd.read_csv(q_path))
        
    dataframe = pd.concat(dataframe, axis=0, ignore_index=True)
    dataframe["id"] = dataframe["File Name"].apply(extract_index).astype(int)
    dataframe = dataframe.set_index("id")
    dataframe = dataframe.sort_index()
    
    return dataframe
    
def merge_results_with_source(source_dataframe, results_dataframe):
    source_dataframe = source_dataframe[~source_dataframe.correct]
    source_dataframe = source_dataframe.set_index("id")
    source_dataframe = source_dataframe.sort_index()
    dataframe = pd.concat([source_dataframe, results_dataframe], axis=1)

    # We want to keep the information about which submission is which 
    dataframe = dataframe.reset_index(drop=False)

    # Can decide between the two...
    # dataframe = dataframe.rename(columns={"Repair": "repair"})
    dataframe = dataframe.rename(columns={"Refactored Correct Code": "repair"})
    dataframe.loc[pd.isnull(dataframe.repair), "repair"] = ""
    dataframe["source_code"] = dataframe["source_code"].apply(simple_clean)
    dataframe["repair"] = dataframe["repair"].apply(simple_clean)

    # renaming of some of the columns, and cleaning the repairs
    columns = list(source_dataframe.columns)
    columns.append("repair")
    columns.append("id")
    dataframe = dataframe[columns]

    return dataframe

def reexecute_repairs(dataframe, grader):
    eval_ds = Dataset.from_pandas(dataframe)
    eval_ds = eval_ds.rename_column("repair", "generation")
    # need the "completion_id" for the evaluation
    eval_ds = eval_ds.add_column("completion_id", [1] * len(dataframe))
    _, eval_ds = evaluate_functional_correctness(eval_ds, grader, k=[1])

    return eval_ds.to_pandas()
    

def extract_index(file_name):
    return int(file_name.split("_")[-1][:-3])
