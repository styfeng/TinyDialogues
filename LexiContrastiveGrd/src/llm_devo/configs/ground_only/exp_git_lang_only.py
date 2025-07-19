import llm_devo.datasets.conceptual_12m as conceptual_12m
from llm_devo.configs.general import\
        add_tk_pad_collate_fn, get_func_for_add_exp_seeds,\
        get_seq_funcs_to_update_key_params, change_max_epoch
import llm_devo.configs.general as cfg_gnrl
import functools
import copy
from itertools import product
import llm_devo.train.tk_funcs as tk_funcs
from transformers import ViTFeatureExtractor
import llm_devo.configs.ground_only.data_func_utils as data_func_utils


tokenizer = tk_funcs.get_git_tokenizer_func()
KWARGS = dict(
        all_things=globals(),
        specify_iter=[],
        specify_epoch=[2, 5, 10],
        model_name='git_large',
        )
DATA_KWARGS = dict(
        max_epochs=10, ckpt_save_interval=10,
        col_name='cc12m',
        tokenizer=tokenizer,
        )

add_exp_seeds = get_func_for_add_exp_seeds(
        DATA_KWARGS, KWARGS)
dino_processor_func = lambda : ViTFeatureExtractor.from_pretrained(
        'facebook/dino-vitb16')

DINO_ARCH_small_KEYS = [
        'base',
        'txt_base',
        'idx_base',
        ]
DINO_ARCH_small_KEYS.extend(data_func_utils.SIMPLE_KEYS_FOR_RW)
MODEL_small_KEYS = [
        'noimg_tie_lyrs_6', # same as noimg_tie
        'dino_cached_50M',
        ]
MAX_EPOCHS_DICT = {
        '100K': 20,
        '500K': 20,
        '1M': 60,
        '5M': 20,
        '15M': 10,
        '50M': 10,
        }
for simple_key, size, model_key in product(
        DINO_ARCH_small_KEYS,
        list(MAX_EPOCHS_DICT.keys()),
        MODEL_small_KEYS):
    exp_prfx = f'{simple_key}_{size}_{model_key}_gitl_s'
    data_func = data_func_utils.get_data_func(
        simple_key,
        size,
        processor_func=dino_processor_func)
    if data_func is None:
        continue
    add_exp_seeds(
            data_func=data_func,
            post_func=get_seq_funcs_to_update_key_params(
                [lambda x: add_tk_pad_collate_fn(x, tokenizer),
                 functools.partial(
                     change_max_epoch,
                     new_max_epochs=MAX_EPOCHS_DICT[size]),
                ]),
            model_name=f'git_{model_key}',
            **cfg_gnrl.get_name_seeds(exp_prfx))
