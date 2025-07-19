import llm_devo.datasets.conceptual_12m as conceptual_12m
from itertools import product
import functools
import copy


DATA_FUNC_ADD_KWARGS = {
        'idx_base': {
            'return_valid_idx': True,
            },
        }

DATA_FUNC_KEY_MAP = {
        'idx_base': 'txt_base',
        }
SIMPLE_KEYS_FOR_RW = []

for key in conceptual_12m.SIMPLE_RW_KEYS:
    DATA_FUNC_ADD_KWARGS[key] = {
            'text_only': False,
            }
    SIMPLE_KEYS_FOR_RW.append(key)
    DATA_FUNC_KEY_MAP['txt_' + key] = key
    SIMPLE_KEYS_FOR_RW.append('txt_' + key)

    now_key = 'idx_' + key
    DATA_FUNC_KEY_MAP[now_key] = key
    DATA_FUNC_ADD_KWARGS[now_key] = {
            'return_valid_idx': True,
            }
    SIMPLE_KEYS_FOR_RW.append(now_key)


def get_data_func(
        simple_key,
        size,
        processor_func,
        **kwargs):
    data_func_add_kwargs = DATA_FUNC_ADD_KWARGS.get(
            simple_key, {})
    if 'processor_func' not in data_func_add_kwargs:
        data_func_add_kwargs['processor_func'] = processor_func
    key_in_data_func = DATA_FUNC_KEY_MAP.get(
            simple_key, simple_key)
    raw_data_func = getattr(
            conceptual_12m, f'get_CC12M_{size}_{key_in_data_func}',
            None)
    if raw_data_func is None:
        return None
    data_func_add_kwargs.update(kwargs)
    return functools.partial(
            raw_data_func,
            **data_func_add_kwargs)
