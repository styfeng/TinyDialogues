import llm_devo.datasets.conceptual_12m as conceptual_12m
from llm_devo.configs.general import\
        add_func_in_general_for_opt, get_general_data_func,\
        add_tk_pad_collate_fn, get_func_for_add_exp_seeds,\
        get_seq_funcs_to_update_key_params, change_max_epoch,\
        change_max_epoch_save_only_final
import llm_devo.configs.general as cfg_gnrl
import functools
import copy
from itertools import product
import llm_devo.train.tk_funcs as tk_funcs
import llm_devo.models.helper as helper
from transformers import ViTFeatureExtractor
import llm_devo.configs.ground_only.data_func_utils as data_func_utils


git_tokenizer = tk_funcs.get_git_tokenizer_func()
KWARGS = dict(
        all_things=globals(),
        specify_iter=[],
        specify_epoch=[2, 5, 10, 20],
        model_name='clip_base-patch16',
        )
DATA_KWARGS = dict(
        max_epochs=20, ckpt_save_interval=15,
        col_name='cc12m',
        tokenizer=git_tokenizer,
        )

add_exp_seeds = get_func_for_add_exp_seeds(
        DATA_KWARGS, KWARGS)

dino_processor_func = lambda : ViTFeatureExtractor.from_pretrained(
        'facebook/dino-vitb16')

BATCH_SIZE_to_GIT_POST_FUNCs = {}
for batch_size in [128, 256, 512, 1024]:
    BATCH_SIZE_to_GIT_POST_FUNCs[batch_size] = cfg_gnrl.get_seq_funcs_to_update_key_params([
            functools.partial(
                add_tk_pad_collate_fn,
                tokenizer=git_tokenizer,
                ),
            cfg_gnrl.get_func_for_replace_batch_processor_to_func(
                helper.clip_loss_processor,
                ),
            cfg_gnrl.get_func_for_change_batch_size(batch_size),
            ])


DINO_BS512_KEYS = [
        'base',
        'idx_base',
        ]
DINO_BS512_KEYS.extend(data_func_utils.SIMPLE_KEYS_FOR_RW)
MAX_EPOCHS_DICT = {
        '100K': 20,
        '500K': 20,
        '1M': 60,
        '5M': 20,
        '15M': 10,
        '50M': 10,
        }
DINO_ARCH_KEYS = [
        'cached_git_like_np1_clip',
        'lcg_ly6_git_like_clip',
        ]

for simple_key, size, dino_key, batch_size in product(
        DINO_BS512_KEYS,
        ['100K', '500K', '1M', '5M', '15M', '50M'],
        DINO_ARCH_KEYS,
        [128, 256, 512, 1024]):
    exp_prfx = f'{simple_key}_bs{batch_size}_{size}_{dino_key}_s'
    model_name = dino_key
    data_func = data_func_utils.get_data_func(
        simple_key,
        size,
        processor_func=dino_processor_func)
    if data_func is None:
        continue
    added_kwargs = {}
    if 'git_like' in dino_key:
        now_base_func = BATCH_SIZE_to_GIT_POST_FUNCs[batch_size]
    now_max_epoch = MAX_EPOCHS_DICT[size]
    add_exp_seeds(
            exp_names=[
                f'{exp_prfx}1',
                f'{exp_prfx}2',
                f'{exp_prfx}3',
                f'{exp_prfx}11',
                f'{exp_prfx}12',
                f'{exp_prfx}21',
                f'{exp_prfx}22',
                ],
            seeds=[1, 2, 3, 11, 12, 21, 22],
            data_func=data_func,
            model_name=model_name,
            post_func=get_seq_funcs_to_update_key_params(
                [now_base_func,
                 functools.partial(
                     change_max_epoch, 
                     new_max_epochs=now_max_epoch),
                ]),
            **added_kwargs)
