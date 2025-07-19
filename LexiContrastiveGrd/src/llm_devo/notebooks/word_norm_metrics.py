import pickle
import os
import ipdb
from itertools import product
from scipy.stats import linregress
import copy
import numpy as np
from tqdm import tqdm
import llm_devo.notebooks.utils as utils
from llm_devo.env_vars import ROOT_DIR_FREQ_ORG
from llm_devo.utils.word_related import load_aoa_data

RESULT_DIR = os.path.join(ROOT_DIR_FREQ_ORG, 'llm_devo_word_norm_results')

class WordNormResults:
    def __init__(
            self, model_name, verbose=False,
            task_name='buchanan_aoa10'):
        self.model = model_name
        self.task_name = task_name
        self.verbose = verbose
        self.load_raw_results()

    def load_raw_results(self):
        pkl_path = os.path.join(
                RESULT_DIR, self.task_name, f'{self.model}.pkl')
        self.raw_data = utils.general_load_raw_results(pkl_path)

    def get_aggre_perf(
            self,
            which_ckpt=None):
        best_perf = 0
        final_test_perf = 0
        best_layer = None

        if which_ckpt is not None:
            search_ckpts = [which_ckpt,]
        else:
            search_ckpts = list(self.raw_data.keys())
        for ckpt in search_ckpts:
            all_data = self.raw_data[ckpt]['all_perfs']
            for layer_idx, rec in enumerate(all_data):
                if np.mean(rec['val']) > best_perf:
                    final_test_perf = np.mean(rec['test'])
                    best_perf = np.mean(rec['val'])
                    best_layer = layer_idx
                if self.verbose:
                    print(f'{layer_idx} {np.mean(rec["val"])}')
            if self.verbose:
                print(f'Best layer {best_layer} for model {self.model}')
        return final_test_perf, None
