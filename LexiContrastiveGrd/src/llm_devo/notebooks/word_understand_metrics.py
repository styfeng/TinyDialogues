import pickle
import os
import copy
import numpy as np
import csv
from tqdm import tqdm
import pdb

import llm_devo.notebooks.utils as nb_utils
from llm_devo.env_vars import ROOT_DIR_FREQ_ORG


class WordUnderstandMetrics:
    RESULT_FOLDER = 'llm_devo_word_understand_results'
    def __init__(
            self, model_name, CACHE_DICT,
            allowed_targets=None,
            allowed_distractors=None,
            task_name='pair_sent',
            just_use_cache=False,
            ):
        self.model_name = model_name
        self.allowed_targets = allowed_targets
        self.allowed_distractors = allowed_distractors
        self.task_name = task_name
        if not just_use_cache:
            self.load_raw_data()
        else:
            self.raw_data = None
        if model_name in CACHE_DICT:
            self.cached_perf = CACHE_DICT[model_name]['cached_perf']
            #if just_use_cache or (len(self.raw_data) == len(self.cached_perf)):
            if just_use_cache:
                self.best_ckpt = CACHE_DICT[model_name]['best_ckpt']
                #print(f'{model_name}, {self.best_ckpt}')
            else:
                self.get_best_ckpt()
                self.dump_cache(CACHE_DICT)
        else:
            self.cached_perf = dict()
            self.get_best_ckpt()
            self.dump_cache(CACHE_DICT)

    def dump_cache(self, CACHE_DICT):
        CACHE_DICT[self.model_name] = {}
        CACHE_DICT[self.model_name]['cached_perf'] = self.cached_perf
        CACHE_DICT[self.model_name]['best_ckpt'] = self.best_ckpt

    def set_result_path(self):
        self.result_path = os.path.join(
                ROOT_DIR_FREQ_ORG, self.RESULT_FOLDER,
                self.task_name, f'{self.model_name}.pkl')

    def load_raw_data(self):
        self.set_result_path()
        self.raw_data = nb_utils.general_load_raw_results(self.result_path)

    def get_bootstrap_stderr(
            self,
            bootstrap_times=10000):
        # Not used anymore
        return None

    def get_bootstrap_stderr_deprecated(
            self,
            bootstrap_times=10000):
        acc = self.correct_trials / self.all_num_trials
        all_accs = []
        for _ in range(bootstrap_times):
            now_sample = np.random.rand(self.all_num_trials)
            all_accs.append(np.sum(now_sample < acc) / self.all_num_trials)
        stderr = np.std(all_accs)
        return stderr

    def get_target(self, name):
        return name.split('_')[0]

    def get_distractor(self, name):
        return name.split('_')[-1]

    def filter_by_targets(self, name, targets):
        target = self.get_target(name)
        if target in targets:
            return True
        else:
            return False

    def filter_by_distractors(self, name, distractors):
        distractor = self.get_distractor(name)
        if distractor in distractors:
            return True
        else:
            return False

    def get_cache_key(
            self, ckpt, targets, distractors,
            normalizer=None):
        key = [ckpt]
        if targets is None:
            key.append(())
        else:
            key.append(tuple(targets))
        if distractors is None:
            key.append(())
        else:
            key.append(tuple(distractors))
        if normalizer is not None:
            key.append(normalizer.model_name)
        return tuple(key)

    def get_better_trial_masks(
            self, likelihood1, likelihood2, mask=None):
        l1_arr = np.asarray([x[0] for x in likelihood1])
        l2_arr = np.asarray([x[0] for x in likelihood2])
        if mask is not None:
            l1_arr = l1_arr[mask]
            l2_arr = l2_arr[mask]
        return l1_arr > l2_arr

    def update_one_name_perf(self, name, normalizer):
        if normalizer is None:
            num_trials = len(self.now_results[name]['likelihood1'])
            self.correct_trials += int(num_trials * self.now_results[name]['acc'])
        else:
            if name not in self.mask_results:
                return
            mask_correct = self.get_better_trial_masks(
                    self.mask_results[name]['likelihood1'],
                    self.mask_results[name]['likelihood2'])
            if len(mask_correct) != len(self.now_results[name]['likelihood1']):
                return
            num_trials = np.sum(mask_correct)
            self.correct_trials += np.sum(self.get_better_trial_masks(
                    self.now_results[name]['likelihood1'],
                    self.now_results[name]['likelihood2'],
                    mask=mask_correct))
        self.all_num_trials += num_trials

    def get_aggre_perf(
            self, ckpt=None, filter_func=None,
            targets=None, distractors=None,
            normalizer=None,
            which_ckpt=None):
        if which_ckpt is not None:
            ckpt = which_ckpt
        if ckpt is None:
            #print(self.model_name, self.best_ckpt)
            ckpt = self.best_ckpt
        self.now_ckpt = ckpt
        self.correct_trials = 0
        self.all_num_trials = 0
        if filter_func is None:
            cache_key = self.get_cache_key(
                    ckpt, targets, distractors, normalizer)
            if cache_key in self.cached_perf:
                return self.cached_perf[cache_key]
        else:
            cache_key = None
        self.now_results = self.raw_data[ckpt]['results']

        if normalizer is not None:
            self.mask_results = normalizer.raw_data['pretrained']['results']

        for name in self.now_results:
            if filter_func is not None:
                should_stay = filter_func(name)
                if not should_stay:
                    continue
            if targets is not None:
                should_stay = self.filter_by_targets(name, targets)
                if not should_stay:
                    continue
            if distractors is not None:
                should_stay = self.filter_by_distractors(
                        name, distractors)
                if not should_stay:
                    continue
            self.update_one_name_perf(name, normalizer)
        acc = self.correct_trials / self.all_num_trials
        stderr = self.get_bootstrap_stderr()
        if cache_key is not None:
            self.cached_perf[cache_key] = (acc, stderr)
        return acc, stderr

    def get_all_target_words(self):
        all_target_words = []
        if self.raw_data is not None:
            now_results = self.raw_data[self.best_ckpt]['results']
            for name in now_results:
                target = self.get_target(name)
                if target not in all_target_words:
                    all_target_words.append(target)
        else:
            for key in self.cached_perf:
                targets = key[1]
                if len(targets) == 1:
                    all_target_words.append(targets[0])
            all_target_words = np.unique(all_target_words)
        return all_target_words

    def get_per_target_word_perf(self):
        all_target_words = self.get_all_target_words()
        target_word_perf = dict()
        for target in all_target_words:
            target_word_perf[target] = self.get_aggre_perf(targets=[target])
        return target_word_perf

    def get_all_distractor_words(self):
        all_distractor_words = []
        now_results = self.raw_data[self.best_ckpt]['results']
        for name in now_results:
            distractor = self.get_distractor(name)
            if distractor not in all_distractor_words:
                all_distractor_words.append(distractor)
        return all_distractor_words

    def get_per_distractor_word_perf(self):
        all_distractor_words = self.get_all_distractor_words()
        distractor_word_perf = dict()
        for distractor in all_distractor_words:
            distractor_word_perf[distractor] = self.get_aggre_perf(distractors=[distractor])
        return distractor_word_perf

    def prefilter_data(self):
        if self.raw_data is None:
            print(f'{self.model_name} not finished in the cache!')
            raise NotImplementedError
        for ckpt in self.raw_data:
            now_results = self.raw_data[ckpt]['results']
            all_names = list(now_results.keys())
            for name in all_names:
                needed = True
                if (self.allowed_targets is not None)\
                        and (self.get_target(name) not in self.allowed_targets):
                    needed = False
                if (self.allowed_distractors is not None)\
                        and (self.get_distractor(name) not in self.allowed_distractors):
                    needed = False
                if not needed:
                    now_results.pop(name)
            self.raw_data[ckpt]['results'] = now_results

    def get_best_ckpt(self):
        best_ckpt = None
        best_perf = None
        self.prefilter_data()
        for ckpt in self.raw_data:
            overall_perf, _ = self.get_aggre_perf(ckpt)
            if (best_perf is None) or (overall_perf > best_perf):
                best_perf = overall_perf
                best_ckpt = ckpt
        self.best_ckpt = best_ckpt


class AggWordUnderstandMetrics:
    def __init__(
            self,
            *args, **kwargs):
        self.all_tasks = [
                'pair_sent',
                'adj_pair_sent',
                'verb_pair_sent',
                ]
        self.all_sub_classes = {}
        for _task in self.all_tasks:
            now_cache = nb_utils.update_word_udr_cache({}, _task)
            self.all_sub_classes[_task] = WordUnderstandMetrics(
                    CACHE_DICT=now_cache,
                    *args, **kwargs)

    def get_aggre_perf(
            self, which_ckpt,
            *args, **kwargs):
        assert which_ckpt != None
        all_perfs = []
        for _task in self.all_tasks:
            all_perfs.append(
                    self.all_sub_classes[_task].get_aggre_perf(
                        which_ckpt=which_ckpt,
                        *args, **kwargs)[0])
        ave_perf = np.mean(all_perfs)
        return ave_perf, None
