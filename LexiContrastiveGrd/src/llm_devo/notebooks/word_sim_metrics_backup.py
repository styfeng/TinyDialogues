import pickle
import os
import copy
import numpy as np
import ipdb
import csv
from tqdm import tqdm
from scipy.stats import linregress
from scipy.stats import spearmanr, pearsonr
import pandas as pd
from itertools import product

import llm_devo
import llm_devo.notebooks.utils as utils
from llm_devo.utils.word_related import load_aoa_data
from llm_devo.env_vars import ROOT_DIR as NESE_ROOT
from llm_devo.env_vars import ROOT_DIR_FREQ_ORG

#NESE_ROOT = '/nese/mit/group/evlab/u/chengxuz'


class WordSimMetrics:
    RESULT_FOLDER = 'llm_devo_word_sim_results'
    def __init__(
            self, model_name,
            task_name='human_sim',
            select_key='res_w_b',
            CACHE_DICT=None,
            just_use_cache=False,
            ):
        self.model_name = model_name
        self.task_name = task_name
        self.select_key = select_key
        self.CACHE_DICT = CACHE_DICT
        self.raw_data = None
        self.just_use_cache = just_use_cache

    def set_result_path(self):
        self.result_path = os.path.join(
                ROOT_DIR_FREQ_ORG, self.RESULT_FOLDER,
                self.task_name, f'{self.model_name}.pkl')

    def load_raw_data(self):
        self.set_result_path()
        self.raw_data = utils.general_load_raw_results(self.result_path)

    def get_ckpt_perf(self):
        cache_key = self.get_cache_key()
        cache_key += f' at ckpt {self.curr_ckpt}'
        if self.CACHE_DICT is not None\
                and cache_key in self.CACHE_DICT:
            return self.CACHE_DICT[cache_key]

        agg_perf = []
        now_data = self.raw_data[self.curr_ckpt]
        for task in self.tasks:
            if task not in now_data:
                return None
            agg_perf.append(
                    [per_layer_data[0]
                     for per_layer_data in now_data[task][self.select_key]])
        agg_perf = np.asarray(agg_perf)
        agg_perf = np.mean(agg_perf, axis=0)
        ret_perf = np.max(agg_perf)
        self.CACHE_DICT[cache_key] = ret_perf
        return ret_perf

    def get_cache_key(self):
        return f'{self.model_name} at high-level task {self.task_name} of tasks {tuple(self.tasks)}'

    def dump_cache(self, CACHE_DICT):
        for key in self.CACHE_DICT:
            if key not in CACHE_DICT:
                CACHE_DICT[key] = self.CACHE_DICT[key]

    def get_best_perf(
            self, tasks,
            which_ckpt=None):
        self.tasks = tasks

        cache_key = self.get_cache_key()
        if which_ckpt is not None:
            cache_key += f' at ckpt {which_ckpt}'

        if self.CACHE_DICT is not None:
            if cache_key in self.CACHE_DICT:
                if which_ckpt is None:
                    return self.CACHE_DICT[cache_key]
                else:
                    return self.CACHE_DICT[cache_key], which_ckpt
        if self.just_use_cache:
            return
        if self.raw_data is None:
            self.load_raw_data()
        if (len(self.raw_data) < 5) and ('pretrained' not in self.model_name):
            #print(f'{self.model_name} likely not finished!')
            pass
        if which_ckpt is None:
            best_ckpt = None
            best_perf = None
            for _ckpt in self.raw_data:
                self.curr_ckpt = _ckpt
                _perf = self.get_ckpt_perf()
                if _perf is None:
                    continue
                if (best_perf is None) or (_perf > best_perf):
                    best_perf = _perf
                    best_ckpt = _ckpt
            if self.CACHE_DICT is not None:
                self.CACHE_DICT[cache_key] = best_perf, best_ckpt
            return best_perf, best_ckpt
        else:
            self.curr_ckpt = which_ckpt
            _perf = self.get_ckpt_perf()
            if self.CACHE_DICT is not None:
                self.CACHE_DICT[cache_key] = _perf
            return _perf, which_ckpt

    def get_aggre_perf(self, *args, **kwargs):
        return self.get_best_perf(*args, **kwargs)


def load_check_sep_results(task, from_task):
    check_result_dir = os.path.join(
            ROOT_DIR_FREQ_ORG,
            'llm_devo_word_sim_collect_related')
    fpath = os.path.join(
            check_result_dir,
            task, f'{from_task}.txt')
    with open(fpath, 'r') as fin:
        all_lines = fin.readlines()
    all_idxs = [int(_line.split(' ')[0]) for _line in all_lines]
    all_diffs = [float(_line.split(' ')[1].strip()) for _line in all_lines]
    return np.asarray(all_idxs), np.asarray(all_diffs)


TASK_FILTER_CACHE = dict()

class FilterWordSimMetrics(WordSimMetrics):
    def __init__(
            self, to_compare='human', 
            aoa_age=10, add_simlex_sep=False,
            add_other_sep=[],
            add_check_sep=[],
            verbose=False,
            *args, **kwargs):
        self.to_compare = to_compare
        self.aoa_age = aoa_age
        self.valid_idxs = None
        self.add_simlex_sep = add_simlex_sep
        self.add_other_sep = add_other_sep
        self.add_check_sep = add_check_sep
        self.verbose = verbose
        super().__init__(*args, **kwargs)

    def subselect_by_idx(self, raw_dict, idxs):
        new_dict = {}
        for key, value in raw_dict.items():
            new_dict[key] = [value[idx] for idx in idxs]
        return new_dict

    def add_three_simple_sep(self):
        if self.add_simlex_sep:
            self.add_sub_tasks.update({
                    'simlex999_adj': {
                        'from_task': 'simlex999',
                        'idxs': list(range(0, 111)),
                        },
                    'simlex999_noun': {
                        'from_task': 'simlex999',
                        'idxs': list(range(111, 777)),
                        }, 
                    'simlex999_verb': {
                        'from_task': 'simlex999',
                        'idxs': list(range(777, 999)),
                        },
                    })

    def add_all_other_sep(self):
        if len(self.add_other_sep) > 0:
            import llm_devo.word_sim.utils as word_sim_utils
            lable_runner = word_sim_utils.PairWordsPOSLabeler(
                    words_of_interest='lex_norm_sim',
                    pos_corpus='coca_fic')
            for other_sep in self.add_other_sep:
                if len(other_sep) == 2:
                    task, label = other_sep
                    name = f'{task}_{label}'
                    special_resample = None
                elif len(other_sep) == 3:
                    task, label, special_resample = other_sep
                    name = f'{task}_{label}_{special_resample}'
                else:
                    raise NotImplementedError
                self.add_sub_tasks[name]\
                        = {'from_task': task, 
                           'idxs': lable_runner.get_idxs_for_label(
                               task, label,
                               special_resample=special_resample)}

    def add_all_check_sep(self):
        if len(self.add_check_sep) > 0:
            for each_check_sep in self.add_check_sep:
                if len(each_check_sep) == 3:
                    from_task, task,\
                            (sta_idxs, end_idxs) = each_check_sep
                    all_idxs, _ = load_check_sep_results(task, from_task)
                    self.add_sub_tasks[f'{from_task}_{task}_{sta_idxs}to{end_idxs}']\
                            = {'from_task': from_task,
                                'idxs': all_idxs[sta_idxs : end_idxs],
                              }
                elif len(each_check_sep) == 4:
                    from_task, task,\
                            (sta_idxs, end_idxs),\
                            sp_setting = each_check_sep
                    all_idxs, _ = load_check_sep_results(task, from_task)
                    curr_idxs = all_idxs[sta_idxs : end_idxs]
                    new_idxs = []
                    new_rels = []
                    reltd = self.string_lists_sims[from_task]['relatedness']
                    if sp_setting == 'match_val':
                        for _idx in curr_idxs:
                            now_r = reltd[_idx]
                            idx_to_add = None
                            for an_idx in range(len(reltd)):
                                if an_idx in new_idxs:
                                    continue
                                if an_idx in curr_idxs:
                                    continue
                                if idx_to_add is None:
                                    idx_to_add = an_idx
                                elif np.abs(reltd[an_idx] - now_r)\
                                     < np.abs(reltd[idx_to_add] - now_r):
                                    idx_to_add = an_idx
                                elif np.abs(reltd[an_idx] - now_r)\
                                     == np.abs(reltd[idx_to_add] - now_r):
                                    if np.random.rand() < 0.5:
                                        idx_to_add = an_idx
                            new_idxs.append(idx_to_add)
                            new_rels.append(reltd[idx_to_add])
                        print(np.std(new_rels))
                    self.add_sub_tasks[f'{from_task}_{task}_{sta_idxs}to{end_idxs}_{sp_setting}']\
                            = {'from_task': from_task,
                                'idxs': new_idxs,
                              }
                else:
                    raise NotImplementedError

    def load_human_data(self):
        from llm_devo.word_sim.eval_word_sim import load_human_sim_data, HUMAN_CSVs
        self.string_lists_sims = load_human_sim_data(
                data_folder=os.path.join(
                    llm_devo.__path__[0], 'word_sim/data'))
        self.datasets = copy.copy(HUMAN_CSVs)
        self.add_sub_tasks = {}
        self.add_three_simple_sep()
        self.add_all_other_sep()
        self.add_all_check_sep()

        for sub_task, sub_task_config in self.add_sub_tasks.items():
            self.string_lists_sims[sub_task] = self.subselect_by_idx(
                    self.string_lists_sims[sub_task_config['from_task']],
                    sub_task_config['idxs'])
            self.datasets.append(sub_task)

    def find_best_layer_to_human(self, tmp_raw_data, _task):
        best_layer = None
        best_corr = None
        for layer in range(len(tmp_raw_data[_task][self.select_key])):
            now_corr = spearmanr(
                    self.string_lists_sims[_task]['relatedness'],
                    tmp_raw_data[_task][self.select_key][layer][1])
            now_corr = now_corr[0]
            if (best_corr is None) or (now_corr > best_corr):
                best_layer = layer
                best_corr = now_corr
        return best_layer

    def replace_human_data(self, tmp_raw_data):
        for _task in tmp_raw_data:
            if False:
                if '_clip_' in self.to_compare:
                    # taking the last layer for clip
                    self.string_lists_sims[_task]['relatedness'] = tmp_raw_data[_task][self.select_key][-1][1]
                else:
                    # taking the first layer for other models
                    self.string_lists_sims[_task]['relatedness'] = tmp_raw_data[_task][self.select_key][0][1]

            best_layer = self.find_best_layer_to_human(
                    tmp_raw_data, _task)
            #print(_task, best_layer, self.to_compare)
            self.string_lists_sims[_task]['relatedness']\
                    = tmp_raw_data[_task][self.select_key][best_layer][1]

    def replace_raw_data_with_best_layer(self):
        for epoch in self.raw_data:
            tmp_raw_data = self.raw_data[epoch]
            for _task in tmp_raw_data:
                best_layer = self.find_best_layer_to_human(
                        tmp_raw_data, _task)
                #print(self.model_name, _task, best_layer, epoch)
                self.raw_data[epoch][_task][self.select_key]\
                        = self.raw_data[epoch][_task][self.select_key][best_layer:best_layer+1]

    def load_raw_data(self):
        super().load_raw_data()
        self.load_human_data()
        if self.to_compare != 'human':
            self.replace_raw_data_with_best_layer()
            model_to_compare, epoch = self.to_compare.split('@')
            tmp_result_path = os.path.join(
                    ROOT_DIR_FREQ_ORG, self.RESULT_FOLDER,
                    self.task_name, f'{model_to_compare}.pkl')
            tmp_raw_data = pickle.load(open(
                tmp_result_path, 'rb'))
            if epoch in tmp_raw_data:
                tmp_raw_data = tmp_raw_data[epoch]
            self.replace_human_data(tmp_raw_data)

        if self.aoa_age is not None:
            self.load_aoa_data()

    def get_cache_key(self):
        cache_key = super().get_cache_key()
        if self.aoa_age is not None:
            cache_key += f' at Aoa age {self.aoa_age}'
        if self.to_compare != 'human':
            cache_key += f' to {self.to_compare}'
        return cache_key

    def load_aoa_data(self):
        global TASK_FILTER_CACHE
        aoa_dict = load_aoa_data()

        self.valid_idxs = dict()
        for task in self.datasets:
            rep_key = f'{task}_atAoA_{self.aoa_age}'
            if not rep_key in TASK_FILTER_CACHE:
                words_left = self.string_lists_sims[task]['words_left']
                words_right = self.string_lists_sims[task]['words_right']
                valid_idx = list(range(len(words_left)))
                for idx, (_left, _right) in enumerate(zip(words_left, words_right)):
                    if (aoa_dict.get(_left, 1000) > self.aoa_age)\
                            or (aoa_dict.get(_right, 1000) > self.aoa_age):
                        valid_idx.remove(idx)
                TASK_FILTER_CACHE[rep_key] = valid_idx
                #print(f'{task} at AoA {self.aoa_age} has {len(valid_idx)} words.')
            else:
                valid_idx = TASK_FILTER_CACHE[rep_key]
            self.valid_idxs[task] = copy.copy(valid_idx)

    def get_data_all_layers(self, task, now_data):
        data_all_layers = []
        corrs_all_layers = []
        if task in self.add_sub_tasks:
            now_model_data = now_data[\
                    self.add_sub_tasks[task]['from_task']][self.select_key]
        else:
            if task not in now_data:
                return None
            now_model_data = now_data[task][self.select_key]
        for per_layer_data in now_model_data:
            _now_corrs = per_layer_data[1]
            if task in self.add_sub_tasks:
                _now_corrs = np.asarray(_now_corrs)[
                        self.add_sub_tasks[task]['idxs']]
            _now_human_corrs = self.string_lists_sims[task]['relatedness']
            if self.valid_idxs is not None:
                _valid_idx = self.valid_idxs[task]
                _now_corrs = np.asarray(_now_corrs)[_valid_idx]
                _now_human_corrs = np.asarray(_now_human_corrs)[_valid_idx]
            if np.any(np.isnan(_now_corrs)):
                good_idx = ~np.isnan(_now_corrs)
                _now_corrs = np.asarray(_now_corrs)[good_idx]
                _now_human_corrs = np.asarray(_now_human_corrs)[good_idx]
            corr = spearmanr(_now_corrs, _now_human_corrs)
            data_all_layers.append(corr[0])
            corrs_all_layers.append(
                    (_now_corrs, _now_human_corrs))
        return data_all_layers, corrs_all_layers

    def get_ckpt_perf(self):
        cache_key = self.get_cache_key()
        cache_key += f' at ckpt {self.curr_ckpt}'
        if self.CACHE_DICT is not None\
                and cache_key in self.CACHE_DICT:
            return self.CACHE_DICT[cache_key]

        agg_perf = []
        now_data = self.raw_data[self.curr_ckpt]
        for task in self.tasks:
            data_all_layers, _ = self.get_data_all_layers(
                    task, now_data)
            agg_perf.append(data_all_layers)
        agg_perf = np.asarray(agg_perf)
        agg_perf = np.mean(agg_perf, axis=0)
        if self.verbose:
            print(self.model_name, agg_perf, self.tasks)
        ret_perf = np.max(agg_perf)
        if self.CACHE_DICT is not None:
            self.CACHE_DICT[cache_key] = ret_perf
        return ret_perf

    def get_best_corrs_for_ckpt_task(
            self, curr_ckpt, task,
            return_rank=True,
            word_pair_filter_func=None):
        self.load_raw_data()
        now_data = self.raw_data[curr_ckpt]
        data_all_layers, corrs_all_layers = self.get_data_all_layers(
                task, now_data)
        best_layer = np.argmax(data_all_layers)
        words_left = self.string_lists_sims[task]['words_left']
        words_right = self.string_lists_sims[task]['words_right']
        words_in_pair = [
                (_left, _right)
                for _left, _right in zip(words_left, words_right)]
        valid_idxs = self.valid_idxs[task]
        words_in_pair = [
                words_in_pair[_idx]
                for _idx in valid_idxs]
        model_corrs, human_corrs = corrs_all_layers[best_layer]
        if word_pair_filter_func is not None:
            now_valid_idx = []
            for _idx in range(len(words_in_pair)):
                if word_pair_filter_func(words_in_pair[_idx]):
                    now_valid_idx.append(_idx)
            words_in_pair = [
                    words_in_pair[_idx]
                    for _idx in now_valid_idx]
            model_corrs = [
                    model_corrs[_idx]
                    for _idx in now_valid_idx]
            human_corrs = [
                    human_corrs[_idx]
                    for _idx in now_valid_idx]
        if return_rank:
            model_corrs_rank = np.argsort(np.argsort(model_corrs))
            human_corrs_rank = np.argsort(np.argsort(human_corrs))
            return (model_corrs_rank, human_corrs_rank), words_in_pair
        else:
            model_corrs = (model_corrs - np.mean(model_corrs)) / np.std(model_corrs)
            human_corrs = (human_corrs - np.mean(human_corrs)) / np.std(human_corrs)
            return (model_corrs, human_corrs), words_in_pair
