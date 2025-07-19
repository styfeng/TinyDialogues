import sklearn
import ipdb
import copy
import argparse
import functools
import os
import re
import numpy as np
import pickle
from tqdm import tqdm

import llm_devo.datasets.lm_eval_word_understd as lm_eval_word_understd
from llm_devo.analysis.get_target_candidates_from_wordbank\
        import get_wdbank_adjs, get_wdbank_verbs, get_wdbank_nouns
import llm_devo.analysis.use_lm_eval as use_lm_eval
from llm_devo.env_vars import ROOT_DIR_FREQ_ORG

RESULT_DIR = os.path.join(
        ROOT_DIR_FREQ_ORG,
        'llm_devo_word_understand_results')


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
            '--targets', default=None, type=str, action='store',
            help='Multiple words can be split by ","')
    parser.add_argument(
            '--high_level_task', default='pair_sent', 
            type=str, action='store')
    parser.add_argument(
            '--extra_forward_mode', default=None, 
            type=str, action='store')
    parser.add_argument(
            '--which_ckpt', default=None,
            type=str, action='store')
    return parser


class WordUnderstandRunner(use_lm_eval.LMEvalRunner):
    def __init__(self, args, result_dir=RESULT_DIR):
        args.fewshot = 0
        self.args = args
        self.result_dir = result_dir
        self.get_task_dict()
        self.get_key_params()
        self.get_lm_model()
        self.get_all_ckpts()
        self.update_all_ckpts()

    def get_task_dict(self):
        task_kwargs = {}
        high_level_task = self.args.high_level_task
        splits = high_level_task.split('_')
        sta_idx = 0
        if splits[0] == 'verb':
            words = get_wdbank_verbs()
            sta_idx = 1
        elif splits[0] == 'adj':
            words = get_wdbank_adjs()
            sta_idx = 1
        else:
            words = get_wdbank_nouns()
        task_kwargs = dict(
                target_words=words, distractor_words=words,
                )

        end_idx = len(splits)
        self.task_dict = lm_eval_word_understd.get_keepall_tasks(**task_kwargs)

    def get_task_in_res(self):
        return self.args.high_level_task

    def filter_finished_tasks(
            self, task_dict, exist_results):
        if exist_results is None:
            return task_dict

        exist_keys = []
        for key in task_dict:
            if key in exist_results['results']:
                exist_keys.append(key)
        for key in exist_keys:
            task_dict.pop(key)
        return task_dict

    def update_exist_results(self, results, exist_results):
        if exist_results is None:
            return results
        for key in exist_results:
            exist_results[key].update(results[key])
        return exist_results

    def eval_all(self):
        args = self.args

        self.get_result_path()
        result_path = self.result_path

        if (not args.overwrite) and (os.path.exists(result_path)):
            exist_results = pickle.load(open(result_path, 'rb'))
        else:
            exist_results = None

        if not args.all_ckpts:
            task_dict = copy.copy(self.task_dict)
            task_dict = self.filter_finished_tasks(task_dict, exist_results)
            results = self.do_one_eval(task_dict)
            if result_path is not None:
                os.system('mkdir -p ' + os.path.dirname(result_path))
                exist_results = self.update_exist_results(
                        results, exist_results)
                pickle.dump(exist_results, open(result_path, 'wb'))
        else:
            now_results = {}
            os.system('mkdir -p ' + os.path.dirname(result_path))
            if (not args.overwrite) and (os.path.exists(result_path)):
                now_results = pickle.load(open(result_path, 'rb'))

            for _ckpt in tqdm(self.all_ckpts):
                if _ckpt in now_results:
                    exist_results = now_results[_ckpt]
                else:
                    exist_results = None
                self.curr_ckpt = _ckpt
                task_dict = copy.copy(self.task_dict)
                task_dict = self.filter_finished_tasks(task_dict, exist_results)
                if len(task_dict) == 0:
                    continue
                self.lm.load_ckpt(os.path.join(self.exp_folder, _ckpt))
                results = self.do_one_eval(task_dict)
                now_results[_ckpt] = self.update_exist_results(
                        results, exist_results)
                pickle.dump(now_results, open(result_path, 'wb'))
        print(f'Results saved to {result_path}')


def main():
    parser = get_parser()
    args = parser.parse_args()

    runner = WordUnderstandRunner(args)
    runner.eval_all()


if __name__ == '__main__':
    main()
