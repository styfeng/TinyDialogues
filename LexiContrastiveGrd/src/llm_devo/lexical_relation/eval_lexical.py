import sklearn
import ipdb
import pandas as pd
import copy
import argparse
import functools
import os
import re
import numpy as np
import pickle
from tqdm import tqdm
import logging
from itertools import chain, product, combinations
from multiprocessing import Pool
import pickle

import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier

import llm_devo.word_sim.eval_word_sim as eval_word_sim
import llm_devo.lexical_relation.data as lex_data
from llm_devo.utils.word_related import load_aoa_data
from llm_devo.env_vars import ROOT_DIR_FREQ_ORG, DEBUG
from llm_devo.notebooks.lexical_metrics import LexicalResults #ADDED BY STEVEN

#BELOW RESULT_DIR CODE CHANGED BY STEVEN

'''
RESULT_DIR = os.path.join(
        ROOT_DIR_FREQ_ORG,
        'llm_devo_lexical_relation_results')
'''

current_directory = os.getcwd()
print(current_directory)

RESULT_DIR = os.path.join(
        current_directory,
        'llm_devo_lexical_relation_results')

def get_parser():
    parser = argparse.ArgumentParser(
            description='Get evaluation metrics through lm_eval')
    parser.add_argument(
            '--ckpt_path', default=None, type=str, action='store')
    parser.add_argument(
            '--setting', default=None, type=str, action='store')
    parser.add_argument(
            '--pretrained', default=None, type=str, action='store')
    parser.add_argument(
            '--all_ckpts', default=False, action='store_true')
    parser.add_argument(
            '--overwrite', default=False, action='store_true')
    parser.add_argument(
            '--high_level_task', default='bcekr',
            type=str, action='store')
    parser.add_argument(
            '--aoa_thres', default=10,
            type=int, action='store')
    parser.add_argument(
            '--dataset', default='CogALexV',
            type=str, action='store')
    parser.add_argument(
            '--which_ckpt', default=None,
            type=str, action='store')
    parser.add_argument(
            '--run_mode', default=None,
            type=str, action='store')
    return parser

DEFAULT_all_embd_methods = [
        'diff', #'cat', 
        #'cat+dot', 'diff+dot',
        'manual_prompt_a',
        ]
class Evaluate:
    manual_prompts = {
            'a': 'Today, I finally discovered the relation between {h} and {t} : {t} is {h}’s ',
            'b': 'Today, I finally discovered the relation between {h} and {t} : ',
            'c': 'I wasn’t aware of this relationship, but I just read in the encyclopedia that {t} is {h}’s ',
            }
    def __init__(self,
                 dataset,
                 label_dict,
                 num_layers=13,
                 target_relation=None,
                 default_config: bool = False,
                 config=None,
                 all_embd_methods=DEFAULT_all_embd_methods):
        self.dataset = dataset
        self.label_dict = label_dict
        self.target_relation = target_relation
        self.all_embd_methods = all_embd_methods
        if default_config:
            self.configs = [dict(
                MLP_config={'random_state': 0},
                embd_method='diff',
                layer_idx=0,
                )]
        elif config is not None:
            self.configs = [config]
        else:
            #learning_rate_init = [0.001, 0.0001]
            #hidden_layer_sizes = [100, 150]

            learning_rate_init = [0.001]
            hidden_layer_sizes = [100]
            self.configs = [dict(
                MLP_config={
                    'random_state': 0, 
                    'learning_rate_init': i[0], 
                    'hidden_layer_sizes': i[1]},
                embd_method=i[2],
                layer_idx=i[3],
                )\
                for i in list(product(
                    learning_rate_init, hidden_layer_sizes,
                    self.all_embd_methods,
                    list(range(num_layers))))]
            new_configs = []
            for _cfg in self.configs:
                m = _cfg['embd_method']
                l = _cfg['layer_idx']
                if (not m.startswith('manual_prompt_')) and (l % 2==1):
                    continue
                if (m.startswith('manual_prompt_')) and (l < num_layers // 2):
                    continue
                new_configs.append(_cfg)
            self.configs = new_configs

    def run_test(self, clf, x, y, per_class_metric: bool = False):
        """ run evaluation on valid or test set """
        y_pred = clf.predict(x)
        p_mac, r_mac, f_mac, _ = precision_recall_fscore_support(y, y_pred, average='macro')
        p_mic, r_mic, f_mic, _ = precision_recall_fscore_support(y, y_pred, average='micro')
        accuracy = sum([a == b for a, b in zip(y, y_pred.tolist())]) / len(y_pred)
        tmp = {
            'accuracy': accuracy,
            'f1_macro': f_mac,
            'f1_micro': f_mic,
            'p_macro': p_mac,
            'p_micro': p_mic,
            'r_macro': r_mac,
            'r_micro': r_mic
        }
        if per_class_metric and self.target_relation is not None:
            for _l in self.target_relation:
                if _l in self.label_dict:
                    p, r, f, _ = precision_recall_fscore_support(y, y_pred, labels=[self.label_dict[_l]])
                    tmp['f1/{}'.format(_l)] = f[0]
                    tmp['p/{}'.format(_l)] = p[0]
                    tmp['r/{}'.format(_l)] = r[0]
        return tmp

    @property
    def config_indices(self):
        return list(range(len(self.configs)))

    def get_dataset_x_y(self, key, config):
        x, y = self.dataset[key]
        x = x[config['embd_method']][config['layer_idx']]
        return x, y

    def __call__(self, config_id, run_mode=None):
        config = self.configs[config_id]
        report = dict()
        # train
        x, y = self.get_dataset_x_y('train', config)
        clf = MLPClassifier(**config['MLP_config']).fit(x, y)
        report.update({'classifier_config': clf.get_params()})
        report.update({'search_config': config.copy()})
        # test
        x, y = self.get_dataset_x_y('test', config)
        tmp = self.run_test(clf, x, y, per_class_metric=True)
        tmp = {'test/{}'.format(k): v for k, v in tmp.items()}
        if run_mode == 'store_full_results':
            y_pred = clf.predict(x)
            tmp['test/y_prediction'] = y_pred
            tmp['test/y_truth'] = y
        report.update(tmp)
        if 'val' in self.dataset:
            x, y = self.get_dataset_x_y('val', config)
            tmp = self.run_test(clf, x, y, per_class_metric=True)
            tmp = {'val/{}'.format(k): v for k, v in tmp.items()}
            if run_mode == 'store_full_results':
                y_pred = clf.predict(x)
                tmp['val/y_prediction'] = y_pred
                tmp['val/y_truth'] = y
            report.update(tmp)
        if DEBUG:
            ipdb.set_trace()
        return report


class LexicalRunner(eval_word_sim.WordSimRunner):
    def __init__(self, args, result_dir=RESULT_DIR):
        self.hidden_states_pos = 2
        self.has_eos_token = False
        self.args = args
        self.result_dir = result_dir
        self.batch_size = 16
        self.load_datasets()
        self.get_key_params()
        self.get_lm_model()
        self.get_all_ckpts()
        self.update_all_ckpts()
        self.update_has_eos_token()
        self.get_all_embd_methods()

    def get_all_embd_methods(self):
        args = self.args
        if args.high_level_task == 'bcekr':
            self.all_embd_methods = DEFAULT_all_embd_methods
        elif args.high_level_task == 'bcekr_cat':
            self.all_embd_methods = [
                    'cat', 
                    'cat+dot', 'diff+dot',
                    ]
        else:
            raise NotImplementedError
        if getattr(self.lm, 'tokenizer', None) is None:
            # special situation for glove and other word models
            self.all_embd_methods = ['diff',]

    def get_task_in_res(self):
        task_in_res = self.args.high_level_task
        if self.args.aoa_thres is not None:
            task_in_res += f'_aoa{self.args.aoa_thres}'
        if self.args.run_mode is not None:
            task_in_res += f'_rm{self.args.run_mode}'
        return task_in_res

    def load_bcekr(self):
        self.datasets = lex_data.get_lexical_relation_data()

    def filter_datasets_by_aoa(self):
        aoa_thres = self.args.aoa_thres
        aoa_dict = load_aoa_data()
        for dataset in self.datasets:
            for split_key in self.datasets[dataset]:
                if split_key == 'label':
                    continue
                now_data = copy.deepcopy(
                        self.datasets[dataset][split_key])
                all_words = now_data['x']
                valid_idx = []
                for idx, (x0, x1) in enumerate(all_words):
                    if (aoa_dict.get(x0, 1000) > aoa_thres)\
                            or (aoa_dict.get(x1, 1000) > aoa_thres):
                        continue
                    valid_idx.append(idx)
                new_data = dict(
                        x=[now_data['x'][_idx] for _idx in valid_idx],
                        y=[now_data['y'][_idx] for _idx in valid_idx],
                        )
                self.datasets[dataset][split_key] = new_data

    def load_datasets(self):
        high_level_task = self.args.high_level_task
        if high_level_task in ['bcekr', 'bcekr_cat']:
            self.load_bcekr()
        else:
            raise NotImplementedError
        if self.args.aoa_thres is not None:
            self.filter_datasets_by_aoa()

    def get_word_embds(self, x_tuple):
        x_left = [_x[0] for _x in x_tuple]
        x_right = [_x[1] for _x in x_tuple]
        self.embd_left = self.get_embedding_list(
                x_left, add_top_blank=True,
                has_eos_token=self.has_eos_token,
                )
        self.embd_right = self.get_embedding_list(
                x_right, add_top_blank=True,
                has_eos_token=self.has_eos_token,
                )

    def get_model_inputs_prompt(self, x_tuple, prompt):
        tokenizer = self.lm.tokenizer
        all_input_ids = []
        all_attention_masks = []
        for now_tuple in x_tuple:
            input_str = [prompt.format(h=now_tuple[0], t=now_tuple[1])]
            if self.args.pretrained == 'gpt2':
                input_str = [tokenizer.bos_token + input_str[0]]
            inputs = tokenizer(
                    input_str, return_tensors="pt",
                    add_special_tokens=True)
            all_input_ids.append(inputs.input_ids)
            all_attention_masks.append(inputs.attention_mask)
        return all_input_ids, all_attention_masks

    def encode_tuple(self, x_tuple):
        if self.curr_embd_method == 'diff':
            final_embd = [
                    _left - _right\
                    for _left, _right in zip(self.embd_left, self.embd_right)]
        elif self.curr_embd_method == 'cat':
            final_embd = [
                    np.concatenate([_left, _right], axis=-1)\
                    for _left, _right in zip(self.embd_left, self.embd_right)]
        elif self.curr_embd_method == 'cat+dot':
            final_embd = [
                    np.concatenate([_left, _right, _left * _right], axis=-1)\
                    for _left, _right in zip(self.embd_left, self.embd_right)]
        elif self.curr_embd_method == 'diff+dot':
            final_embd = [
                    np.concatenate([_left - _right, _left * _right], axis=-1)\
                    for _left, _right in zip(self.embd_left, self.embd_right)]
        elif self.curr_embd_method.startswith('manual_prompt_'):
            self.extraction_id = -1
            prompt = Evaluate.manual_prompts[self.curr_embd_method[-1]]
            input_ids, attention_masks = self.get_model_inputs_prompt(
                    x_tuple, prompt)
            final_embd = self.get_all_embeddings(
                    input_ids, attention_masks)
        else:
            raise NotImplementedError('Embd method not recognized')
        return final_embd

    def do_curr_dataset(self):
        self.curr_data = copy.deepcopy(self.datasets[self.curr_dataset])
        label_dict = self.curr_data.pop('label')
        self.dataset_for_Evaluator = {}
        num_layers = None
        for _k, _v in self.curr_data.items():
            x_tuple = [tuple(_x) for _x in _v['x']]
            all_embds = {}
            self.get_word_embds(x_tuple)
            for embd_method in self.all_embd_methods:
                self.curr_embd_method = embd_method
                embds = self.encode_tuple(x_tuple)
                all_embds[embd_method] = embds
                num_layers = len(embds)
            self.dataset_for_Evaluator[_k] = [all_embds, _v['y']]

        evaluator = Evaluate(
                self.dataset_for_Evaluator, label_dict, 
                num_layers=num_layers,
                all_embd_methods=self.all_embd_methods)
        report = []
        for idx in tqdm(evaluator.config_indices, desc=self.curr_dataset):
            now_res = evaluator(
                    idx,
                    run_mode=self.args.run_mode)
            report.append(now_res)
        return report

    def do_one_eval(self, results):
        if (self.args.dataset is not None) and (self.args.dataset != 'None'):
            now_dataset = self.args.dataset
            assert now_dataset in self.datasets
            datasets_to_work = [now_dataset,]
        else:
            datasets_to_work = list(self.datasets.keys())
        for which_ds in tqdm(datasets_to_work, desc='All datasets'):
            if which_ds in results:
                continue
            self.curr_dataset = which_ds
            results[which_ds] = self.do_curr_dataset()
            if self.args.run_mode == 'store_full_results':
                results[f'{which_ds}_raw'] = self.datasets[which_ds]
        return results

    #BELOW FUNCTION ADDED BY STEVEN
    def eval_all(self):
        results = {}
        results = self.do_one_eval(results)
        return results
        
    #BELOW FUNCTION ADDED BY STEVEN
    def aggregate_results(self, results):
        """Aggregate results from multiple configurations."""
        best_perf = 0
        final_test_perf = 0
        for dataset, result_list in results.items():
            for result in result_list:
                if f'val/f1_macro' in result:
                    if result[f'val/f1_macro'] > best_perf:
                        best_perf = result[f'val/f1_macro']
                        final_test_perf = result[f'test/f1_macro']
        return final_test_perf

    #BELOW FUNCTION ADDED BY STEVEN
    def do_one_eval(self, results):
        if (self.args.dataset is not None) and (self.args.dataset != 'None'):
            now_dataset = self.args.dataset
            assert now_dataset in self.datasets
            datasets_to_work = [now_dataset,]
        else:
            datasets_to_work = list(self.datasets.keys())
        for which_ds in tqdm(datasets_to_work, desc='All datasets'):
            if which_ds in results:
                continue
            self.curr_dataset = which_ds
            results[which_ds] = self.do_curr_dataset()
            if self.args.run_mode == 'store_full_results':
                results[f'{which_ds}_raw'] = self.datasets[which_ds]
        return results

def main():
    parser = get_parser()
    args = parser.parse_args()

    runner = LexicalRunner(args)
    results = runner.eval_all()

    final_test_perf = runner.aggregate_results(results) #ADDED BY STEVEN
    print(f"Final aggregated test performance (f1_macro): {final_test_perf}") #ADDED BY STEVEN
    
    #pkl_path = os.path.join(RESULT_DIR, self.task_name, f'{self.model}.pkl')
    #test = LexicalResults(model_name='pretrained/gpt2') #Expected: 0.49363410418274023
    #test = LexicalResults(model_name='miniBERTa/test_train_GPT2-small_tinydialogue_reversed_20n_1e-04')
    #print(test.get_aggre_perf()) #ADDED BY STEVEN

if __name__ == '__main__':
    main()
