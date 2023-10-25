""" 
Utilities for creating training and validation datasets 
for program repair when having access to correct and incorrect
solutions but no ground truth 
"""

import os
import pandas as pd
from tqdm import tqdm
from src.utils.data_structures import find_closest_in_dict
from src.utils.distance import rougelcsum_dist
from src.utils.code import clean_code, simple_clean
from src.utils.files import create_dir, json2data, save_json 


class Mapping():

    def __init__(self, name, config, dataset_handler) -> None:
        self.name = name
        self.config = config
        self.dataset_handler = dataset_handler

        self._init_directories()
        self._load_or_create_mapping()


    def _init_directories(self):
        # load the mapping if it exists
        base_path = self.dataset_handler.config.path
        self.save_dir = os.path.join(base_path, "cache", self.name)
        create_dir(self.save_dir)
        self.save_dir = os.path.join(self.save_dir, "mapping.json")

    def _load_or_create_mapping(self):
        self.mapping = {}
        if os.path.exists(self.save_dir) and not self.config.remap:
            self.mapping = json2data(self.save_dir)
        else:
            self.mapping = self._map()
            save_json(self.mapping, self.save_dir)

    def _map(self):
        # possibly doesn't have it 
        train_df = self.dataset_handler.get_split("train")
        val_df = self.dataset_handler.get_split("val")
        # possibly if we ask twice for the same dataset... (e.g. singapore)
        df = pd.concat([train_df, val_df], axis=0).drop_duplicates("id")
        assert df.set_index("id").index.is_unique

        mapping = self._get_mapping(df)
        
        return map_incorrect_to_closest_correct(df, 
                                                rougelcsum_dist,
                                                mapping)

    def _get_mapping(self, df):
        """ Overriden by subclass. """
        return {}
    
    def apply_mapping_to_dataset(self, dataset):
        dataset = dataset.map(add_correction_to_incorrect, 
                              fn_kwargs={"mapping": self.mapping}) 

        return dataset 


class DefaultMapping(Mapping):
    def __init__(self, config, dataset_handler) -> None:
        super().__init__("default", config, dataset_handler)


def add_correction_to_incorrect(example, mapping):
    """ Add to each training element, the associated correction. """
    
    code = example["source_code"]
    
    rp = mapping.get(code, "")
    if not rp:
        sc = simple_clean(code)
        rp = mapping.get(sc, "")
    if not rp:
        sc = clean_code(code)
        rp = mapping.get(sc, "")
    if not rp:
        rp = find_closest_in_dict(mapping, code)

    # get the closest to the repair

    example["repair"] = rp

    return example


def map_incorrect_to_closest_correct(df, dist_f, mapping={}):
    groups = df.groupby("problem_id").groups
    for _, index in tqdm(groups.items()):
        subset = df.loc[index]
        
        correct_df = subset[subset.correct].set_index("id")
        incorrect_df = subset[~subset.correct].set_index("id")
        
        if not len(correct_df) or not len(incorrect_df):
            continue
            
        for incor in incorrect_df["source_code"]:
            if incor in mapping: continue
            distances = [(cor, dist_f(incor, cor))
                         for _, cor in correct_df["source_code"].items()]
            
            cor = sorted(distances, key=lambda p: p[1])[0][0]
            mapping[incor] = cor
            
    return mapping
