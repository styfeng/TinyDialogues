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
import llm_devo

import torch
import numpy as np

import llm_devo.word_sim.eval_word_sim as eval_word_sim
import llm_devo.word_norm.eval_word_norm as eval_word_norm
from llm_devo.utils.word_related import load_aoa_data
from llm_devo.env_vars import ROOT_DIR_FREQ_ORG, DEBUG
from sklearn.svm import LinearSVC

RESULT_DIR = os.path.join(
        ROOT_DIR_FREQ_ORG,
        'llm_devo_pos_pred_results')


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
            '--high_level_task', default='coca_fic',
            type=str, action='store')
    parser.add_argument(
            '--label_method', default='single',
            type=str, action='store')
    parser.add_argument(
            '--words_of_interest', default='lex_norm_sim',
            type=str, action='store')
    parser.add_argument(
            '--aoa_thres', default=10,
            type=int, action='store')
    parser.add_argument(
            '--which_ckpt', default=None,
            type=str, action='store')
    parser.add_argument(
            '--layer_subselect', default=2,
            type=int, action='store')
    parser.add_argument(
            '--run_mode', default=None,
            type=str, action='store')
    return parser


class PosPredRunner(eval_word_sim.WordSimRunner):
    def __init__(self, args, result_dir=RESULT_DIR):
        self.hidden_states_pos = 2
        self.has_eos_token = False
        self.args = args
        self.result_dir = result_dir
        self.batch_size = 16
        self.load_pos_labels()
        self.get_key_params()
        self.get_lm_model()
        self.get_all_ckpts()
        self.update_all_ckpts()
        try:
            empty_tks = self.lm.tokenizer(' ').input_ids[-1]
            self.has_eos_token = self.lm.tokenizer.decode(empty_tks) == self.lm.tokenizer.eos_token
        except:
            pass

    def load_pos_labels(self):
        task = self.args.high_level_task
        raw_pos_labels = pickle.load(
                open(os.path.join(
                    './data', task,
                    self.args.words_of_interest + '.pkl'), 'rb'))
        pos_labels = {}
        for word in raw_pos_labels:
            all_labels = list(raw_pos_labels[word].keys())
            all_labels = sorted(all_labels, key=lambda x: raw_pos_labels[word][x])
            pos_labels[word] = all_labels[-1]
        self.pos_labels = pos_labels

    def get_task_in_res(self):
        args = self.args
        base_name = f'{args.high_level_task}_lbl_{args.label_method}'
        if args.words_of_interest != 'lex_norm_sim':
            base_name += f'_words_{args.words_of_interest}'
        if self.args.aoa_thres != 10:
            base_name += f'_aoa{self.args.aoa_thres}'
        if self.args.run_mode is not None:
            base_name += f'_rm{self.args.run_mode}'
        return base_name

    def get_splits(self):
        num_splits = 4
        split_comps = 10
        np.random.seed(0)
        all_splits = []
        num_words = len(self.all_words)
        #num_words = len(self.embeddings)
        for _ in range(num_splits):
            now_order = np.random.permutation(num_words)
            train_idx = int(num_words\
                            / split_comps\
                            * (split_comps - 2))
            val_idx = int(num_words\
                            / split_comps\
                            * (split_comps - 1))
            train = now_order[:train_idx]
            val = now_order[train_idx:val_idx]
            test = now_order[val_idx:]
            all_splits.append((train, val, test))
        self.all_splits = all_splits

    def get_full_splits(self):
        eval_word_norm.WordNormRunner.get_full_splits(self)

    def get_labels_from_idxs(self, idxs):
        labels = [self.pos_labels[self.all_words[_idx]] for _idx in idxs]
        return labels

    def get_train_X_y(self, embds, idxs):
        labels = self.get_labels_from_idxs(idxs)
        return embds, labels

    def get_perf(self, cls, embds, idxs):
        labels = self.get_labels_from_idxs(idxs)
        return cls.score(embds, labels)

    def run_splits(self):
        all_perfs = []
        all_layer_idxs = list(range(
                0, len(self.embeddings),
                self.args.layer_subselect))
        all_Cs = [1e-2, 1.0, 1e2]
        for layer_idx, now_C in tqdm(
                product(all_layer_idxs, all_Cs),
                desc='Combination', total=len(all_layer_idxs) * len(all_Cs)):
            now_train_perfs = []
            now_val_perfs = []
            now_test_perfs = []
            for train, val, test in self.all_splits:
                train_embds = self.embeddings[layer_idx][train]
                val_embds = self.embeddings[layer_idx][val]
                test_embds = self.embeddings[layer_idx][test]
                cls = LinearSVC(C=now_C)

                X, y = self.get_train_X_y(train_embds, train)
                if DEBUG:
                    ipdb.set_trace()
                cls.fit(X, y)
                train_perf = self.get_perf(
                        cls, train_embds, train)
                val_perf = self.get_perf(
                        cls, val_embds, val)
                test_perf = self.get_perf(
                        cls, test_embds, test)
                print(f'Layer {layer_idx} C {now_C}, train: {train_perf:.3f},'\
                      + f' val: {val_perf:.3f},'\
                      + f' test: {test_perf:.3f}'
                      )
                now_train_perfs.append(train_perf)
                now_val_perfs.append(val_perf)
                now_test_perfs.append(test_perf)
            all_perfs.append({
                'train': now_train_perfs,
                'val': now_val_perfs,
                'test': now_test_perfs,
                '_parameters': (layer_idx, now_C),
                })
        return all_perfs

    def filter_words_by_aoa(self, clean_words, all_words):
        aoa_thres = self.args.aoa_thres
        aoa_dict = load_aoa_data()

        valid_idx = []
        for idx, word in enumerate(clean_words):
            if (aoa_dict.get(word, 1000) > aoa_thres):
                continue
            valid_idx.append(idx)
        clean_words = [
                clean_words[_idx]
                for _idx in valid_idx]
        all_words = [
                all_words[_idx]
                for _idx in valid_idx]
        return clean_words, all_words

    def get_best_paramset_in_earlier_results(self):
        if self.args.pretrained is None:
            if self.args.all_ckpts:
                model_name = os.path.join(
                        self.col_name, f'{self.exp_id}.pkl')
            else:
                model_name = os.path.join(
                        'untrained', self.col_name, f'{self.exp_id}.pkl')
        else:
            model_name = os.path.join(
                    'pretrained',
                    f"{self.args.pretrained.replace('/', '_').replace('-', '_')}.pkl")
        task_name = f'{self.args.high_level_task}'\
                    + f'_lbl_{self.args.label_method}'
        fpath = os.path.join(
                self.result_dir,
                task_name,
                model_name)
        earlier_data = pickle.load(open(fpath, 'rb'))
        if getattr(self, 'curr_ckpt', None) is not None:
            earlier_data = earlier_data[self.curr_ckpt]
        earlier_data = earlier_data['all_perfs']
        best_idx = 0
        for _idx in range(1, len(earlier_data)):
            if np.mean(
                    earlier_data[_idx]['val']\
                    > earlier_data[best_idx]['val']):
                best_idx = _idx
        return earlier_data[best_idx]['_parameters']

    def run_splits_keep_results(self):
        all_perfs = []
        layer_idx, best_C = self.get_best_paramset_in_earlier_results()
        print(f'Looking at layer {layer_idx} with C {best_C}')
        for layer_idx, now_C in [(layer_idx, best_C)]:
            now_train_perfs = []
            now_test_perfs = []
            now_train_results = []
            now_test_results = []
            for train, test in tqdm(self.all_splits, desc='Splits'):
                train_embds = self.embeddings[layer_idx][train]
                test_embds = self.embeddings[layer_idx][test]
                cls = LinearSVC(C=now_C)

                X, y = self.get_train_X_y(train_embds, train)
                cls.fit(X, y)
                train_perf = self.get_perf(
                        cls, train_embds, train)
                test_perf = self.get_perf(
                        cls, test_embds, test)
                print(f'Layer {layer_idx} C {now_C}, train: {train_perf:.3f},'\
                      + f' test: {test_perf:.3f}'
                      )
                now_train_perfs.append(train_perf)
                now_test_perfs.append(test_perf)
                now_train_results.append(
                        (cls.predict(train_embds),
                         self.get_labels_from_idxs(train)))
                now_test_results.append(
                        (cls.predict(test_embds),
                         self.get_labels_from_idxs(test)))
            all_perfs.append({
                'train': now_train_perfs,
                'test': now_test_perfs,
                'train_results': now_train_results,
                'test_results': now_test_results,
                })
        return all_perfs

    def do_one_eval(self, results):
        if 'all_perfs' in results:
            return results
        all_words = list(self.pos_labels.keys())
        words_for_embs = copy.deepcopy(all_words)
        for idx, _word in enumerate(words_for_embs):
            if '_' in _word:
                words_for_embs[idx] = _word.split('_')[0]
        if self.args.aoa_thres is not None:
            words_for_embs, all_words = self.filter_words_by_aoa(
                    words_for_embs, all_words)
        self.all_words = all_words
        embd_list_kwargs = dict(
                has_eos_token=self.has_eos_token,
                add_top_blank=True,
                )
        self.embeddings = self.get_embedding_list(
                words_for_embs, **embd_list_kwargs)

        if self.args.run_mode is None:
            self.get_splits()
            results['all_perfs'] = self.run_splits()
        elif self.args.run_mode == 'store_full_results':
            self.get_full_splits()
            results['all_perfs'] = self.run_splits_keep_results()
            results['all_splits'] = self.all_splits
            results['all_words'] = self.all_words
        else:
            raise NotImplementedError
        return results


def main():
    parser = get_parser()
    args = parser.parse_args()

    runner = PosPredRunner(args)
    runner.eval_all()


if __name__ == '__main__':
    main()
