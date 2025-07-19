import lm_eval
from lm_eval.base import rf, Task
from lm_eval.metrics import mean
import os
import re
import numpy as np
import functools
import pickle
import pdb
from tqdm import tqdm

from nltk.corpus import cmudict
from llm_devo.env_vars import ROOT_DIR_FREQ_ORG, DEBUG
DIST_SENT_DIR = os.path.join(ROOT_DIR_FREQ_ORG, 'dist_sent_results')

class WordUnderstandTask(Task):
    VERSION = 0
    DATASET_PATH = None

    def __init__(
            self, 
            target_word='airplane', distractor_word='dog',
            example_source='yourdict',
            num_sents=40, candidate_model='roberta-large',
            clm_score_model='opt-6.7b',
            data_dir=None, cache_dir=None, download_mode=None):
        self.target_word = target_word
        self.distractor_word = distractor_word
        self.example_source = example_source
        self.num_sents = num_sents
        self.candidate_model = candidate_model
        self.clm_score_model = clm_score_model
        self.download()
        self._training_docs = None
        self._fewshot_docs = None

    def get_sents_for_one_dist_word(self, distractor_word=None):
        if distractor_word is None:
            distractor_word = self.distractor_word

        res_folder = os.path.join(
                DIST_SENT_DIR, 
                f'{self.candidate_model}_clm_{self.clm_score_model}')
        filepath = os.path.join(
                res_folder, self.example_source,
                f'{self.target_word}_to_{distractor_word}.pkl')
        assert os.path.exists(filepath), f'{filepath} does not exist!'
        try:
            sent_res = pickle.load(open(filepath, 'rb'))
        except:
            print(f'{filepath} is wrong!')
            raise NotImplementedError

        best_per_sent_res = []
        for _sent in sent_res:
            if _sent is not None:
                now_score = _sent['top_res'][0][0]\
                        + _sent['raw_loglkhd']\
                        - _sent['raw_w_dst_loglkhd']
                best_per_sent_res.append(
                        ((now_score, _sent['top_res'][0][1]),
                         _sent['sentence']))
        best_per_sent_res = sorted(best_per_sent_res, key=lambda x: -x[0][0])
        sentences_as_list = [
                dict(sentence_good=_sent[1], sentence_bad=_sent[0][1])
                for _sent in best_per_sent_res]
        sentences_as_list = sentences_as_list[:self.num_sents]
        return sentences_as_list

    def download(self):
        self.sentences_as_list = self.get_sents_for_one_dist_word()

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def validation_docs(self):
        # The HF dataset only contains a "train" dataset, but the harness expects a "validation"
        # dataset. Let's use the training dataset, on the assumption that the model wasn't actually
        # trained on this data.
        return self.sentences_as_list

    def evaluation_docs(self):
        # The HF dataset only contains a "train" dataset, but the harness expects a "validation"
        # dataset. Let's use the training dataset, on the assumption that the model wasn't actually
        # trained on this data.
        import datasets
        sentences_as_dict = {'sentence_good': [], 'sentence_bad': []}
        for sent_dict in self.sentences_as_list:
            sentences_as_dict['sentence_good'].append(
                    sent_dict['sentence_good'])
            sentences_as_dict['sentence_bad'].append(
                    sent_dict['sentence_bad'])
        return datasets.Dataset.from_dict(sentences_as_dict)

    def invalid_doc_for_prompt(self, d):
        return False

    def fewshot_context(
        self, doc, num_fewshot, provide_description=None,
        rnd=None, description=None,
        rng=None,
    ):
        assert num_fewshot == 0
        if rng is not None:
            rnd = rng
        assert (
            rnd is not None
        ), "A `random.Random` generator argument must be provided to `rnd`"
        assert not provide_description, (
            "The `provide_description` arg will be removed in future versions. To prepend "
            "a custom description to the context, supply the corresponding string via the  "
            "`description` arg."
        )
        if provide_description is not None:
            # nudge people to not specify it at all
            print(
                "WARNING: provide_description is deprecated and will be removed in a future version in favor of description_dict"
            )

        if not 'for_MLM' in lm_eval.__path__[0]:
            return ""
        else:
            return "", {}

    def doc_to_text(self, doc):
        # this method is invoked by tests only
        return ""

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["sentence_good"] + " " + doc["sentence_bad"]

    def doc_to_target(self, doc):
        # this method is invoked by tests only
        return ""

    def construct_requests(self, doc, ctx, args=None):
        assert not ctx

        # Calculate the loglikelihood for the good and the bad sentence.
        # Note that loglikelihood translates the "" prefix to the "<|endoftext|>" token
        return [
            rf.loglikelihood("", doc["sentence_good"]),
            rf.loglikelihood("", doc["sentence_bad"]),
        ]

    def process_results(self, doc, results):
        likelihood1, likelihood2 = results

        # the model got this case right iff the good sentence scored higher than the bad sentence
        acc = 1.0 if likelihood1 > likelihood2 else 0.0

        return {
            "acc": acc,
        }

    def higher_is_better(self):
        return {
            "acc": True,
        }

    def aggregation(self):
        return {
            "acc": mean,
        }

    def get_logging_info(self):
        return dict(
                target_word=self.target_word,
                distractor_word=self.distractor_word,
                )

        
class GroupWordUnderstandTask(WordUnderstandTask):
    def __init__(self, all_distractors, *args, **kwargs):
        self.all_distractors = all_distractors
        super().__init__(distractor_word=None, *args, **kwargs)

    def download(self):
        self.sentences_as_list = []
        for distractor_word in self.all_distractors:
            self.sentences_as_list.extend(
                    self.get_sents_for_one_dist_word(distractor_word))


class WordUnderstandTaskKeepAll(WordUnderstandTask):
    def process_results(self, doc, results):
        likelihood1, likelihood2 = results

        # the model got this case right iff the good sentence scored higher than the bad sentence
        acc = 1.0 if likelihood1 > likelihood2 else 0.0

        return {
            "acc": acc,
            "likelihood1": likelihood1, 
            "likelihood2": likelihood2,
        }

    def aggregation(self):
        return {
            "acc": mean,
            "likelihood1": lambda x: x,
            "likelihood2": lambda x: x,
        }


class RawModSentLikelihd(WordUnderstandTask):
    def starts_w_vwl_sd(self, word):
        try:
            return self.pronunciations.get(word, [])[0][0][-1].isdigit()
        except:
            return False

    def replace_word_and_trivial_change(self, sent):
        new_sent = sent.replace(self.target_word, self.tmp_distractor_word)
        if self.target_start_w_vwl == self.dist_start_w_vwl:
            return new_sent
        target_pos = sent.find(self.target_word)
        pre_words = re.findall('\w+', sent[:target_pos])
        if len(pre_words) == 0:
            return new_sent
        raw_pre_word = pre_words[-1]
        pre_word = raw_pre_word.strip().lower()
        if pre_word != 'a' and pre_word != 'an':
            return new_sent
        if (pre_word  == 'an') and self.dist_start_w_vwl:
            return new_sent
        if (pre_word  == 'a') and (not self.dist_start_w_vwl):
            return new_sent
        if pre_word == 'an':
            new_word = 'a'
        else:
            new_word = 'an'
        raw_word_start_pos = target_pos-1-len(raw_pre_word)
        raw_word_in_sent = sent[raw_word_start_pos : target_pos-1]
        if raw_word_in_sent != raw_pre_word:
            return new_sent
        if raw_pre_word.strip()[0] == 'A':
            new_word = 'A' + new_word[1:]
        new_sent = new_sent[:raw_word_start_pos] + new_word + new_sent[target_pos-1:]
        return new_sent

    def get_precomputed_path(self):
        res_folder = os.path.join(
                DIST_SENT_DIR, 
                f'{self.candidate_model}_clm_{self.clm_score_model}_rawmod')
        filepath = os.path.join(
                res_folder, self.example_source,
                f'{self.target_word}_to_{self.tmp_distractor_word}.pkl')
        return filepath

    def exist_precomputed(self):
        return os.path.exists(self.get_precomputed_path())

    def load_precomputed_sent_list(self):
        try:
            return pickle.load(open(self.get_precomputed_path(), 'rb'))
        except:
            print(self.get_precomputed_path())
            assert False

    def save_precomputed(self, sent_list):
        path = self.get_precomputed_path()
        path_dir = os.path.dirname(path)
        if not os.path.exists(path_dir):
            os.system('mkdir -p ' + path_dir)
        pickle.dump(sent_list, open(path, 'wb'))
    
    def get_sents_for_one_dist_word(self, distractor_word=None):
        if distractor_word is None:
            distractor_word = self.distractor_word
        self.tmp_distractor_word = distractor_word
        if self.exist_precomputed():
            return self.load_precomputed_sent_list()

        sent_list = super().get_sents_for_one_dist_word(distractor_word)
        self.pronunciations = cmudict.dict()
        self.target_start_w_vwl = self.starts_w_vwl_sd(self.target_word)
        self.dist_start_w_vwl = self.starts_w_vwl_sd(distractor_word)
        for _sent_pair in sent_list:
            _sent_pair['sentence_good'] = self.replace_word_and_trivial_change(
                    _sent_pair['sentence_good'])
            _sent_pair['sentence_bad'] = self.replace_word_and_trivial_change(
                    _sent_pair['sentence_bad'])
        self.save_precomputed(sent_list)
        return sent_list

    def process_results(self, doc, results):
        likelihood1, likelihood2 = results
        return {
            "likelihood1": likelihood1, 
            "likelihood2": likelihood2,
        }

    def aggregation(self):
        return {
            "likelihood1": lambda x: x,
            "likelihood2": lambda x: x,
        }


def build_task_w_params(builder=WordUnderstandTask, *args, **kwargs):
    return builder(*args, **kwargs)


def register_tasks():
    from lm_eval import tasks
    candidate_model = 'roberta-large'
    clm_model = 'opt-6.7b'
    source = 'yourdict'
    res_folder = os.path.join(
            DIST_SENT_DIR, f'{candidate_model}_clm_{clm_model}',
            source)
    all_files = os.listdir(res_folder)
    all_files = list(filter(lambda x: x.endswith('.pkl'), all_files))
    for each_file in all_files:
        splits = each_file[:-4].split('_')
        if len(splits) != 3:
            continue
        if splits[1] != 'to':
            continue
        target, _, distractor = splits

        new_name = f'word_understd_{target}_to_{distractor}'
        tasks.TASK_REGISTRY[new_name]\
                = functools.partial(
                        build_task_w_params,
                        target_word=target, distractor_word=distractor)


def register_fast_group_tasks():
    candidate_model = 'fast_len_lt30_roberta-large'
    clm_model = 'opt-125m'
    source = 'yourdict'
    res_folder = os.path.join(
            DIST_SENT_DIR, f'{candidate_model}_clm_{clm_model}',
            source)
    all_files = os.listdir(res_folder)
    all_files = list(filter(lambda x: x.endswith('.pkl'), all_files))
    all_files = list(sorted(all_files))
    all_targets = dict()
    for each_file in all_files:
        splits = each_file[:-4].split('_')
        if len(splits) != 3:
            continue
        if splits[1] != 'to':
            continue
        target, _, distractor = splits
        if target not in all_targets:
            all_targets[target] = []
        all_targets[target].append(distractor)

    for target_word, all_distractors in all_targets.items():
        new_name = f'word_understd_fast_group_{target_word}'
        tasks.TASK_REGISTRY[new_name]\
                = functools.partial(
                        build_task_w_params,
                        target_word=target_word,
                        all_distractors=all_distractors,
                        builder=GroupWordUnderstandTask,
                        candidate_model=candidate_model,
                        clm_score_model=clm_model,
                        num_sents=20)


def get_keepall_tasks(
        target_words=None,
        builder=None,
        source='yourdict',
        distractor_words=None):
    candidate_model = 'fast_len_lt30_roberta-large'
    clm_model = 'opt-125m'
    res_folder = os.path.join(
            DIST_SENT_DIR, f'{candidate_model}_clm_{clm_model}',
            source)
    all_files = os.listdir(res_folder)
    all_files = list(filter(lambda x: x.endswith('.pkl'), all_files))
    all_files = list(sorted(all_files))

    if builder is None:
        builder = WordUnderstandTaskKeepAll

    tasks = dict()
    for each_file in tqdm(all_files, desc='Get tasks'):
        splits = each_file[:-4].split('_')
        if len(splits) != 3:
            continue
        if splits[1] != 'to':
            continue
        target, _, distractor = splits
        if (target_words is not None) and (target not in target_words):
            continue
        if (distractor_words is not None) and (distractor not in distractor_words):
            continue
        if '*' in distractor:
            continue
        tasks[f'{target}_to_{distractor}'] = builder(
                    target_word=target,
                    distractor_word=distractor,
                    candidate_model=candidate_model,
                    clm_score_model=clm_model,
                    num_sents=20,
                    example_source=source)
    return tasks
