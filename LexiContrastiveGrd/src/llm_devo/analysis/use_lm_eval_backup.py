import torch
import pdb
import argparse
import functools
import os
import ipdb
import re
import numpy as np
import pickle
from tqdm import tqdm

from llm_devo.env_vars\
        import ROOT_DIR, ROOT_DIR_FREQ_ORG

from lm_eval import tasks, evaluator

RESULT_DIR = os.path.join(
        ROOT_DIR,
        'llm_devo_results')
DEFAULT_EXP_FOLDER = os.path.join(
        ROOT_DIR,
        'llm_devo_models')
DEFAULT_EXP_FOLDER_NEW = os.path.join(
        ROOT_DIR_FREQ_ORG,
        'llm_devo_models')


def get_parser():
    parser = argparse.ArgumentParser(
            description='Get evaluation metrics through lm_eval')
    parser.add_argument(
            '--ckpt_path', default=None, type=str, action='store')
    parser.add_argument(
            '--setting', default=None, type=str, action='store')
    parser.add_argument(
            '--task', default='wsc273', type=str, action='store')
    parser.add_argument(
            '--fewshot', default=0, type=int, action='store')
    parser.add_argument(
            '--limit', default=None, type=int, action='store')
    parser.add_argument(
            '--pretrained', default=None, type=str, action='store')
    parser.add_argument(
            '--all_ckpts', default=False, action='store_true')
    parser.add_argument(
            '--overwrite', default=False, action='store_true')
    parser.add_argument(
            '--which_ckpt', default=None, 
            type=str, action='store')
    return parser


class LMEvalRunner:
    def __init__(self, args, result_dir=RESULT_DIR):
        self.args = args
        self.result_dir = result_dir
        self.get_key_params()
        self.get_lm_model()
        self.get_all_ckpts()
        self.update_all_ckpts()

    def get_key_params(self):
        setting = self.args.setting
        if setting is not None:
            from llm_devo.train.utils import get_setting_func
            setting_func = get_setting_func(setting)
            key_params = setting_func({})
        else:
            key_params = {}
        self.key_params = key_params

    def get_lm_model(self):
        from llm_devo.analysis.lm_eval_models import LLMDevoModels
        args = self.args
        self.extra_forward_mode = getattr(
                args, 'extra_forward_mode',
                None)
        pretrained = args.pretrained
        if pretrained is not None:
            if pretrained.startswith('microsoft/git'):
                from lm_eval.models.gpt2 import HFLM
                self.lm = HFLM(
                        pretrained=pretrained,
                        batch_size=16,
                        device='',
                        )
                self.lm.tokenizer.eos_token_id = self.lm.tokenizer.sep_token_id
            elif pretrained.startswith('openai/clip'):
                from llm_devo.analysis.lm_eval_models import AutoHFLM
                from transformers import CLIPTextModel
                self.lm = AutoHFLM(
                        pretrained=pretrained,
                        batch_size=16,
                        device='',
                        builder=CLIPTextModel)
            elif pretrained.startswith('EleutherAI/pythia'):
                act_pretrained, revision = pretrained.split('_')
                from transformers import GPTNeoXForCausalLM, AutoTokenizer
                model = GPTNeoXForCausalLM.from_pretrained(
                  act_pretrained,
                  revision=revision)
                tokenizer = AutoTokenizer.from_pretrained(
                  act_pretrained,
                  revision=revision)
                tokenizer.pad_token = tokenizer.eos_token
                self.lm = LLMDevoModels(
                        model, tokenizer,
                        extra_forward_mode=self.extra_forward_mode)
            elif pretrained == 'facebook/flava-full':
                from transformers import FlavaTextModel, AutoTokenizer
                model = FlavaTextModel.from_pretrained("facebook/flava-full")
                tokenizer = AutoTokenizer.from_pretrained("facebook/flava-full")
                self.lm = LLMDevoModels(
                        model, tokenizer,
                        extra_forward_mode=self.extra_forward_mode)
            elif pretrained == 'facebook/flava-full_mm':
                from transformers import FlavaModel, AutoTokenizer
                model = FlavaModel.from_pretrained("facebook/flava-full")
                model.config.multimodal_config.output_hidden_states = True
                tokenizer = AutoTokenizer.from_pretrained("facebook/flava-full")
                self.lm = LLMDevoModels(
                        model, tokenizer,
                        extra_forward_mode=self.extra_forward_mode)
            else:
                try:
                    from llm_devo.analysis.lm_eval_models import get_model
                    self.lm = get_model(pretrained)(
                            pretrained=pretrained,
                            batch_size=16,
                            device='',
                            )
                except:
                    from lm_eval.models.gpt2 import HFLM
                    self.lm = HFLM(
                            pretrained=pretrained,
                            batch_size=16,
                            device='',
                            )
            return
        assert NotImplementedError

    def get_task_in_res(self):
        now_task = self.now_task
        args = self.args
        task_in_res = now_task
        if args.fewshot > 0:
            task_in_res += f'_s{args.fewshot}'
        if args.limit is not None:
            task_in_res += f'_l{args.limit}'
        return task_in_res

    def get_result_path(self):
        self.has_finished = False
        args = self.args
        task_in_res = self.get_task_in_res()

        if not args.all_ckpts:
            if args.pretrained is None:
                assert getattr(self, 'extra_forward_mode', None) is None
                if args.ckpt_path is not None:
                    result_path = os.path.join(
                            self.result_dir, task_in_res,
                            self.col_name, 
                            f'{self.exp_id}_{os.path.basename(args.ckpt_path)}.pkl')
                else:
                    result_path = os.path.join(
                            self.result_dir, task_in_res, 'untrained',
                            self.col_name, f'{self.exp_id}.pkl'
                            )
                    if (not args.overwrite) and (os.path.exists(result_path)):
                        self.has_finished = True
            else:
                fwd_mode = getattr(self, 'extra_forward_mode', None)
                pret_in_name = args.pretrained.replace('/', '_').replace('-', '_')
                if fwd_mode is None:
                    fname = pret_in_name + '.pkl'
                else:
                    fname = pret_in_name + f'_fwd_{fwd_mode}.pkl'
                result_path = os.path.join(
                        self.result_dir, task_in_res, 'pretrained',
                        fname)
                if (not args.overwrite) and (os.path.exists(result_path)):
                    self.has_finished = True
        else:
            fname = self.exp_id
            if getattr(self, 'extra_forward_mode', None) is not None:
                fname += '_fwd_' + self.extra_forward_mode
            result_path = os.path.join(
                    self.result_dir, task_in_res,
                    self.col_name, f'{fname}.pkl')
        self.result_path = result_path

    def get_task_dict(self):
        if self.now_task.startswith('CBT'):
            from lm_eval.tasks import cbt
            class CBTV(cbt.CBTBase):
                DATASET_NAME = "V"
            class CBTP(cbt.CBTBase):
                DATASET_NAME = "P"
            # CBTCN or CBTNE
            if self.now_task in ['CBTCN', 'CBTNE']:
                self.task_dict = {
                        self.now_task: getattr(cbt, self.now_task)(),
                        }
            elif self.now_task == 'CBTV':
                self.task_dict = {
                        self.now_task: CBTV(),
                        }
            elif self.now_task == 'CBTP':
                self.task_dict = {
                        self.now_task: CBTP(),
                        }
            else:
                raise NotImplementedError
        elif self.now_task.endswith('_aoa10'):
            now_task = self.now_task[:len(self.now_task) - len('_aoa10')]
            now_task = tasks.get_task_dict([now_task])[now_task]
            from llm_devo.nlp_bench.aoa_thres_utils import get_aoa_info
            import datasets
            key_of_interest = 'validation'
            if key_of_interest not in now_task.dataset:
                key_of_interest = 'test'
            aoa_info = get_aoa_info(None, task=now_task, key=key_of_interest)
            select_indxs = [_data for _data in np.where(np.asarray(aoa_info) < 10)[0]]
            new_dataset = datasets.Dataset.from_dict(now_task.dataset[key_of_interest][select_indxs])
            now_task.dataset[key_of_interest] = new_dataset
            self.task_dict = {self.now_task: now_task}
        elif self.now_task in ['winogrande_physical', 'winogrande_social']:
            from llm_devo.nlp_bench.winogrande_related import stim_select
            now_task = stim_select.get_simple_subset(
                    select_type=self.now_task.split('_')[1])
            self.task_dict = {self.now_task: now_task}
        else:
            self.task_dict = tasks.get_task_dict([self.now_task])

    def do_one_eval(self, task_dict=None):
        if task_dict is None:
            task_dict = self.task_dict
        results = evaluator.evaluate(
                lm=self.lm,
                task_dict=task_dict,
                num_fewshot=self.args.fewshot,
                limit=getattr(self.args, 'limit', None),
                bootstrap_iters=100000,
                description_dict=None,
                decontamination_ngrams_path=None,
                )
        return results

    def get_ckpts_from_exp_folder(self, exp_folder):
        if not os.path.exists(exp_folder):
            return []
        all_ckpts = os.listdir(exp_folder)
        all_ckpts = list(filter(lambda x: x.startswith('epoch_') and x.endswith('pth'), all_ckpts))
        return all_ckpts

    def find_top_ckpts_by_PPL(self, top_ckpts=6):
        model_name = f'{self.col_name}/{self.exp_id}'
        if '_sw_' in model_name:
            ppl_task_name = 'sw_p20_st64_len128'
            dataset = 'smashwords'
        elif '_chd_' in model_name:
            ppl_task_name = 'aochildes_st64_len128'
            dataset = 'aochildes'
        elif '_wiki_' in model_name:
            ppl_task_name = 'wiki_p05_st64_len128'
            dataset = 'wiki'
        else:
            return []

        from llm_devo.notebooks import ppl_metrics, utils
        runner = ppl_metrics.PPLResults(
                model_name=model_name,
                task_name=ppl_task_name,
                report_sfx='_validation',
                )
        aggre_kwargs = {'dataset': dataset}
        all_perfs, all_eps = utils.get_epoch_traj_from_runner(
                runner, aggre_kwargs=aggre_kwargs)
        sorted_idxs = np.argsort(all_perfs)
        wanted_eps = [
                f'epoch_{all_eps[_i]}.pth'
                for _i in sorted_idxs[:top_ckpts]]
        return wanted_eps

    def update_all_ckpts(self):
        which_ckpt = getattr(self.args, 'which_ckpt', None)
        if which_ckpt is None:
            return
        if which_ckpt == 'PPL_determined':
            wanted_ckpts = self.find_top_ckpts_by_PPL()
            print(wanted_ckpts)
        else:
            wanted_ckpts = self.args.which_ckpt.split(',')
        self.all_ckpts = list(filter(
                lambda x: x in wanted_ckpts,
                self.all_ckpts))

    def update_has_eos_token(self):
        if getattr(self, 'has_eos_token', None) is None:
            self.has_eos_token = False
        try:
            empty_tks = self.lm.tokenizer(' ').input_ids[-1]
            self.has_eos_token = self.lm.tokenizer.decode(empty_tks) == self.lm.tokenizer.eos_token
        except:
            pass

    def get_all_ckpts(self):
        args = self.args

        key_params = self.key_params
        self.exp_id = key_params.get('exp_id', 'test_train')
        self.col_name = key_params.get('col_name', 'miniBERTa')
        if not args.all_ckpts:
            return
        self.exp_folder = os.path.join(
                DEFAULT_EXP_FOLDER_NEW, self.col_name, self.exp_id)
        all_ckpts = self.get_ckpts_from_exp_folder(self.exp_folder)
        if len(all_ckpts) == 0:
            self.exp_folder = os.path.join(
                    DEFAULT_EXP_FOLDER, self.col_name, self.exp_id)
            all_ckpts = self.get_ckpts_from_exp_folder(self.exp_folder)
        self.all_ckpts = all_ckpts

    def handle_one_task(self, now_task):
        self.now_task = now_task
        args = self.args

        self.get_result_path()
        result_path = self.result_path
        if self.has_finished:
            return
        self.get_task_dict()

        if not args.all_ckpts:
            results = self.do_one_eval()
            print(results)
            if result_path is not None:
                os.system('mkdir -p ' + os.path.dirname(result_path))
                pickle.dump(results, open(result_path, 'wb'))
        else:
            now_results = {}
            os.system('mkdir -p ' + os.path.dirname(result_path))
            if (not args.overwrite) and (os.path.exists(result_path)):
                now_results = pickle.load(open(result_path, 'rb'))

            all_ckpts = list(filter(lambda x: x not in now_results, self.all_ckpts))
            for _ckpt in tqdm(self.all_ckpts):
                self.lm.load_ckpt(os.path.join(self.exp_folder, _ckpt))
                results = self.do_one_eval()
                print(_ckpt)
                print(results)
                now_results[_ckpt] = results
                pickle.dump(now_results, open(result_path, 'wb'))
