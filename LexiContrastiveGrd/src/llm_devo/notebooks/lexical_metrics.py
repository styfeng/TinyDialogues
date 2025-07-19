import pickle
import os
import copy
import numpy as np
from tqdm import tqdm
import llm_devo.notebooks.utils as utils
from llm_devo.env_vars import ROOT_DIR, ROOT_DIR_FREQ_ORG
from llm_devo.utils.word_related import load_aoa_data

#BELOW RESULT_DIR CHANGED BY STEVEN

#RESULT_DIR = os.path.join(ROOT_DIR_FREQ_ORG, 'llm_devo_lexical_relation_results')

current_directory = os.getcwd()
print(current_directory)

RESULT_DIR = os.path.join(
        current_directory,
        'llm_devo_lexical_relation_results')

class LexicalResults:
    def __init__(self, model_name, CACHE_DICT={}, task_name='bcekr_aoa10'):
        self.model = model_name
        self.task_name = task_name
        self.CACHE_DICT = CACHE_DICT
        self.load_raw_results()

    def load_raw_results(self):
        pkl_path = os.path.join(
                RESULT_DIR, self.task_name, f'{self.model}.pkl')
        if pkl_path not in self.CACHE_DICT:
            try:
                with open(pkl_path, 'rb') as file:
                    data = pickle.load(file)
                self.CACHE_DICT[pkl_path] = data
                self.raw_data = self.CACHE_DICT[pkl_path]
            except Exception as e:
                print(f"Error loading pickle file: {e}")
                return
        else:
            self.raw_data = self.CACHE_DICT[pkl_path]
        
        # Debugging information
        print(f"Loaded raw data type: {type(self.raw_data)}")
        if isinstance(self.raw_data, list):
            print(f"First element type in raw data: {type(self.raw_data[0])}")
        
        if isinstance(self.raw_data, dict):
            one_ckpt = list(self.raw_data.keys())[0]
            self.datasets = list(self.raw_data[one_ckpt].keys())
        elif isinstance(self.raw_data, list) and len(self.raw_data) > 0 and isinstance(self.raw_data[0], dict):
            one_ckpt = self.raw_data[0]
            self.datasets = list(one_ckpt.keys())
        else:
            print("Unexpected data format in pickle file.")
            return

    def get_aggre_perf(self, dataset='CogALexV', metric='f1_macro', which_ckpt=None):
        best_perf = 0
        final_test_perf = 0
        best_rec = None

        if which_ckpt is not None:
            search_ckpts = [which_ckpt,]
        else:
            search_ckpts = list(self.raw_data.keys())
        for ckpt in search_ckpts:
            all_data = self.raw_data[ckpt][dataset]
            for rec in all_data:
                if f'val/{metric}' in rec:
                    if rec[f'val/{metric}'] > best_perf:
                        final_test_perf = rec[f'test/{metric}']
                        best_perf = rec[f'val/{metric}']
                        best_rec = rec
                else:
                    if rec[f'test/{metric}'] > final_test_perf:
                        final_test_perf = rec[f'test/{metric}']
                        best_rec = rec
        #print(best_rec['search_config']['embd_method'], best_rec['search_config']['layer_idx'])
        return final_test_perf, None
