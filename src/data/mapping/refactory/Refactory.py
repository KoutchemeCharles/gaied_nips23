""" Using the Refactory automated tool to map incorrect submissions to correct ones. """

import os
import time
import tempfile

from subprocess import call

from src.data.mapping.Mapping import Mapping
from .preparation import (
    create_save_dir, 
    extract_from_function, 
    put_in_function
)
from .analysis import (
    collate_individual_csvs,
    merge_results_with_source, 
    reexecute_repairs
) 

from datasets import Dataset

class Refactory(Mapping):
    """ 
    Using the Refactory automated repair tool
    to create artificial repairs from a dataset. 
    """

    def __init__(self, config, dataset_handler) -> None:
        super().__init__("refactory", config, dataset_handler) 

    def _get_mapping(self, df):
        
        dataset = Dataset.from_pandas(df, preserve_index=False)
        
        if not self.dataset_handler.functional_form:
            dataset = dataset.map(put_in_function)
        
        with tempfile.TemporaryDirectory() as tmpdirname:
            create_save_dir(dataset, tmpdirname)
            run_tool(self.config.tool_path, tmpdirname)
            results_dataframe = collate_individual_csvs(tmpdirname)

        source_dataframe = dataset.to_pandas()    
        dataframe = merge_results_with_source(source_dataframe, 
                                              results_dataframe)
        
        # Unwrap the code from the artificial function if we added them 
        if not self.dataset_handler.functional_form:
            dataframe["repair"] = dataframe.repair.apply(extract_from_function)
            dataframe["source_code"] = dataframe.source_code.apply(extract_from_function)

        # We test the repairs found by the ART to ensure they are indeed correct 
        grade_fn = self.dataset_handler.grade_fn
        dataframe = reexecute_repairs(dataframe, grade_fn)
        
        # Obtain the mapping from incorrect to correct
        pairs = zip(dataframe["source_code"], 
                    dataframe["generation"], 
                    dataframe["generation_correct"])
        mapping = {sc: g for sc, g, c in pairs if c}

        print("Refactory managed to repair", len(mapping), "solutions")
        
        return mapping  
    

def run_tool(tool_path, dir_path):
    start = time.time()

    questions = os.listdir(dir_path)
    args = [
        "python3", f"{tool_path}/run.py",
        f"-d", dir_path, "-q", *questions,
        f"-s", "100", "-f", "-m", "-b", "-c"]
    call(args)
    end = time.time()

    print("Executing refactory toke", end - start, "seconds")


