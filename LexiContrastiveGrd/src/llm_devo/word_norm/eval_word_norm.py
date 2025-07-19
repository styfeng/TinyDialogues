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
from llm_devo.utils.word_related import load_aoa_data
from llm_devo.env_vars import ROOT_DIR_FREQ_ORG, DEBUG
from llm_devo.word_norm.feature_in_context import feature_data
from sklearn.cross_decomposition import PLSRegression

RESULT_DIR = os.path.join(
        ROOT_DIR_FREQ_ORG,
        'llm_devo_word_norm_results')


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
            '--high_level_task', default='buchanan',
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
    parser.add_argument(
            '--embd_proc', default=None, 
            type=str, action='store')
    return parser


def load_buchanan_data():
    data_path = os.path.join(
            llm_devo.__path__[0],
            'word_norm/data/external/buchanan/cue_feature_words.csv')
    return feature_data.BuchananFeatureNorms(
            data_path)

class WordNormRunner(eval_word_sim.WordSimRunner):
    def __init__(self, args, result_dir=RESULT_DIR):
        self.hidden_states_pos = 2
        self.has_eos_token = False
        self.args = args
        self.result_dir = result_dir
        self.batch_size = 16
        self.pls_num_components = 100
        self.load_feature_norms()
        self.get_key_params()
        self.get_lm_model()
        self.get_all_ckpts()
        self.update_all_ckpts()
        try:
            empty_tks = self.lm.tokenizer(' ').input_ids[-1]
            self.has_eos_token = self.lm.tokenizer.decode(empty_tks) == self.lm.tokenizer.eos_token
        except:
            pass

    def load_feature_norms(self):
        task = self.args.high_level_task
        if task == 'buchanan':
            self.feature_norms = load_buchanan_data()
        else:
            raise NotImplementedError

    def get_task_in_res(self):
        task_in_res = self.args.high_level_task
        if self.args.aoa_thres is not None:
            task_in_res += f'_aoa{self.args.aoa_thres}'
        if self.args.run_mode is not None:
            task_in_res += f'_rm{self.args.run_mode}'
        if getattr(
                self.args,
                'embd_proc', None) != None:
            task_in_res += f'_embd{self.args.embd_proc}'
        return task_in_res

    def get_splits(self):
        num_splits = 2
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
        split_comps = 10
        all_splits = []
        num_words = len(self.all_words)
        np.random.seed(0)
        now_order = np.random.permutation(num_words)
        num_words_per_seg = num_words / split_comps
        all_segments = []
        for seg_idx in range(split_comps):
            sta_idx = int(seg_idx * num_words_per_seg)
            end_idx = min(
                    int((seg_idx+1) * num_words_per_seg),
                    num_words)
            now_segment = list(now_order[sta_idx:end_idx])
            all_segments.append(now_segment)

        for sp_idx in range(split_comps):
            train = []
            for tmp_idx in range(split_comps):
                if tmp_idx == sp_idx:
                    continue
                train.extend(all_segments[tmp_idx])
            test = all_segments[sp_idx]
            all_splits.append((train, test))
        self.all_splits = all_splits

    def get_max_k_perf(self, plsr, embds, idxs):
        if len(embds.shape) == 3:
            embds = np.mean(embds, axis=1)
        logits = plsr.predict(embds)
        all_perfs = []
        for _i, _idx in enumerate(idxs):
            word = self.all_words[_idx]
            gold = self.feature_norms.get_feature_vector(word)
            gold_feats = self.feature_norms.get_features(word)
            n = len(gold_feats)
            ind = np.argpartition(logits[_i], -n)[-n:]
            feats = []
            for _feat_i in ind:
                _feat = self.feature_norms.feature_map.get_object(_feat_i)
                feats.append(_feat)
            num_in_top_k = len(set(feats).intersection(set(gold_feats)))
            all_perfs.append(num_in_top_k / n)
        return np.mean(all_perfs)

    def get_norm_from_idx(self, idx):
        word = self.all_words[idx]
        norm = self.feature_norms.get_feature_vector(word)
        norm = [1 if val > 0 else 0 for val in norm]
        return norm

    def get_train_X_y(self, train_embds, train_idxs):
        X = train_embds
        y = []
        for idx in train_idxs:
            norm = self.get_norm_from_idx(idx)
            y.append(norm)
        return X, y

    def run_splits(self):
        all_perfs = []
        for layer_idx in tqdm(
                range(
                    0, len(self.embeddings),
                    self.args.layer_subselect),
                desc='Layers'):
            now_train_perfs = []
            now_val_perfs = []
            now_test_perfs = []
            for train, val, test in self.all_splits:
                train_embds = self.embeddings[layer_idx][train]
                val_embds = self.embeddings[layer_idx][val]
                test_embds = self.embeddings[layer_idx][test]
                plsr = PLSRegression(
                        n_components=self.pls_num_components,
                        scale=False)

                X, y = self.get_train_X_y(train_embds, train)
                if DEBUG:
                    ipdb.set_trace()
                plsr.fit(X, y)
                train_perf = self.get_max_k_perf(
                        plsr, train_embds, train)
                val_perf = self.get_max_k_perf(
                        plsr, val_embds, val)
                test_perf = self.get_max_k_perf(
                        plsr, test_embds, test)
                print(f'Layer {layer_idx}, train: {train_perf:.3f},'\
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

    def get_max_k_perf_with_results(
            self, plsr, embds, idxs):
        if len(embds.shape) == 3:
            embds = np.mean(embds, axis=1)
        logits = plsr.predict(embds)
        all_perfs = []
        all_results = []
        for _i, _idx in enumerate(idxs):
            word = self.all_words[_idx]
            gold = self.feature_norms.get_feature_vector(word)
            gold_feats = self.feature_norms.get_features(word)
            n = len(gold_feats)
            ind = np.argpartition(logits[_i], -n)[-n:]
            feats = []
            for _feat_i in ind:
                _feat = self.feature_norms.feature_map.get_object(_feat_i)
                feats.append(_feat)
            num_in_top_k = len(set(feats).intersection(set(gold_feats)))
            all_perfs.append(num_in_top_k / n)
            all_results.append(
                    (set(feats), set(gold_feats)))
        return np.mean(all_perfs), all_results

    def get_best_layer_idx_in_earlier_results(self):
        model_name = os.path.join(
                self.col_name, f'{self.exp_id}.pkl')
        fpath = os.path.join(
                self.result_dir,
                f'{self.args.high_level_task}_aoa{self.args.aoa_thres}',
                model_name)
        earlier_data = pickle.load(open(fpath, 'rb'))
        if getattr(self, 'curr_ckpt', None) is not None:
            earlier_data = earlier_data[self.curr_ckpt]
        earlier_data = earlier_data['all_perfs']
        layer_idx = 0
        for _idx in range(1, len(earlier_data)):
            if np.mean(
                    earlier_data[_idx]['val']\
                    > earlier_data[layer_idx]['val']):
                layer_idx = _idx
        return layer_idx

    def run_splits_keep_results(self):
        all_perfs = []
        all_layers = list(range(
                0, len(self.embeddings),
                self.args.layer_subselect))
        if len(all_layers) > 1:
            all_layers = [all_layers[self.get_best_layer_idx_in_earlier_results()]]
        print(f'Looking at layer {all_layers[0]}')
        # just run best layer
        for layer_idx in all_layers:
            now_train_perfs = []
            now_test_perfs = []
            now_train_results = []
            now_test_results = []
            for train, test in tqdm(self.all_splits, desc='All splits'):
                train_embds = self.embeddings[layer_idx][train]
                test_embds = self.embeddings[layer_idx][test]
                plsr = PLSRegression(
                        n_components=self.pls_num_components,
                        scale=False)

                X, y = self.get_train_X_y(train_embds, train)
                if DEBUG:
                    ipdb.set_trace()
                plsr.fit(X, y)
                train_perf, train_results = self.get_max_k_perf_with_results(
                        plsr, train_embds, train)
                test_perf, test_results = self.get_max_k_perf_with_results(
                        plsr, test_embds, test)
                print(f'Layer {layer_idx}, train: {train_perf:.3f},'\
                      + f' test: {test_perf:.3f}'
                      )
                now_train_perfs.append(train_perf)
                now_test_perfs.append(test_perf)
                now_train_results.append(train_results)
                now_test_results.append(test_results)
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
        all_words = list(self.feature_norms.vocab.keys())
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
        embd_proc = getattr(self.args, 'embd_proc', None)
        if embd_proc is not None:
            if embd_proc == 'L2norm':
                for idx, embd in enumerate(self.embeddings):
                    now_norm = np.linalg.norm(embd, axis=-1)
                    new_embd = embd / now_norm[:, np.newaxis]
                    self.embeddings[idx] = new_embd
            else:
                raise NotImplementedError

        if self.args.run_mode is None:
            self.get_splits()
            results['all_perfs'] = self.run_splits()
        elif self.args.run_mode == 'store_full_results':
            self.get_full_splits()
            results['all_perfs'] = self.run_splits_keep_results()
            results['all_splits'] = self.all_splits
            results['feature_norms'] = self.feature_norms
        else:
            raise NotImplementedError
        return results


def main():
    parser = get_parser()
    args = parser.parse_args()

    runner = WordNormRunner(args)
    runner.eval_all()


if __name__ == '__main__':
    main()
