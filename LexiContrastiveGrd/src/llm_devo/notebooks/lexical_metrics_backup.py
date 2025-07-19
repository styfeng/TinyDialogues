import pickle
import os
import copy
import numpy as np
from tqdm import tqdm
import llm_devo.notebooks.utils as utils
from llm_devo.env_vars import ROOT_DIR, ROOT_DIR_FREQ_ORG
from llm_devo.utils.word_related import load_aoa_data
RESULT_DIR = os.path.join(ROOT_DIR_FREQ_ORG, 'llm_devo_lexical_relation_results')


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
            data = pickle.load(open(pkl_path, 'rb'))
            self.CACHE_DICT[pkl_path] = data
            self.raw_data = self.CACHE_DICT[pkl_path]
        else:
            self.raw_data = self.CACHE_DICT[pkl_path]
        if os.path.basename(os.path.dirname(pkl_path)) == 'pretrained':
            self.raw_data = utils.naive_expand(
                    ['pretrained',],
                    self.raw_data)
        if os.path.basename(
                os.path.dirname(os.path.dirname(
                    pkl_path))) == 'untrained':
            self.raw_data = utils.naive_expand(
                    ['untrained',],
                    self.raw_data)
        one_ckpt = list(self.raw_data.keys())[0]
        self.datasets = list(self.raw_data[one_ckpt].keys())

    def get_aggre_perf(
            self, dataset='CogALexV',
            metric='f1_macro',
            which_ckpt=None):
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
