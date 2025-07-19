import os
import pdb
import setuptools
import torch
import ipdb
import copy
import functools
from itertools import product

from transformers import OPTForCausalLM
from transformers.models.opt.modeling_opt import OPTConfig
from .git import get_dino_git_func,\
        get_noimg_more_lyrs_git_func, get_dino_cached_git_func
from .multi_comb import get_multi_comb_model
from .flamingo_hf_opt import get_dino_flamingo_model_func
from .clip import get_cached_clip_git_like_func
DEBUG = int(os.environ.get(
        'DEBUG',
        '0')) == 1


def get_opt_func(opt_model_size='125m'):
    model_name = f"facebook/opt-{opt_model_size}"
    config = OPTConfig.from_pretrained(model_name)
    model = OPTForCausalLM(config=config)
    return model


def clip_loss_processor(
        model, loss_func, data_batch):
    if 'labels' in data_batch:
        data_batch.pop('labels')
    model_outputs = model(
            return_loss=True, **data_batch)
    return {'loss': model_outputs['loss']}


MODEL_NAME_TO_FUNC = {
        'git_noimg_tie_lyrs_6': functools.partial(
            get_noimg_more_lyrs_git_func,
            num_layers=6,
            output_with_bias=False,
            tie_output=True),
        'git_dino_cached_50M': functools.partial(
            get_dino_cached_git_func,
            hidden_states_fname='conceptual_12m.get_CC12M_50M_base.pkl'),
        'lcg_ly6_git_like_clip': functools.partial(
            get_cached_clip_git_like_func,
            text_max='sep_crt_ip_sp2k_w_causal_d3v1_git_ly5',
            change_txt_builder='GITEncoder_AttWd',
            text_hid_layers=1),
        'cached_git_like_np1_clip': get_cached_clip_git_like_func,
        }


cmb_exp_suffix_to_mixw_map = {
        '_1v1': [1, 1],
        '_1v2': [1, 2],
        '_1vd5': [1, 0.5],
        '_1vd25': [1, 0.25],
        '_1vd125': [1, 0.125],
        '_1vd0625': [1, 0.0625],
        '_1vd03125': [1, 0.03125],
        }
cmb_exp_name_to_func_map = {
        }

for _exp_name, _exp_suffix in product(
        list(cmb_exp_name_to_func_map.keys()),
        list(cmb_exp_suffix_to_mixw_map.keys())):
    MODEL_NAME_TO_FUNC[f'cmb_{_exp_name}{_exp_suffix}'] = functools.partial(
            cmb_exp_name_to_func_map[_exp_name],
            mix_weights=cmb_exp_suffix_to_mixw_map[_exp_suffix])
