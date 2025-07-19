import llm_devo.models.helper as helper
from llm_devo.train.tk_funcs import get_tokenizer_func
import ipdb
from llm_devo.train.tk_funcs import tk_pad_collate_fn
import functools
from transformers import DataCollatorForLanguageModeling
from itertools import product
import copy


def add_func_in_general(
        func_name,
        data_func,
        exp_name=None,
        seed=None,
        model_name=None,
        all_things=None,
        post_func=None,
        **kwargs):

    if exp_name is None:
        exp_name = func_name
    def _func(key_params):
        key_params = data_func(key_params)
        if model_name == '350m':
            key_params['get_model_func'] = functools.partial(
                    helper.get_opt_func, 
                    opt_model_size='350m')
        elif model_name in helper.MODEL_NAME_TO_FUNC:
            key_params['get_model_func'] = helper.MODEL_NAME_TO_FUNC[model_name]
        elif model_name is not None:
            raise NotImplementedError
        key_params['exp_id'] = exp_name
        key_params['seed'] = seed
        key_params.update(kwargs)
        if post_func is not None:
            key_params = post_func(key_params)
        return key_params

    if all_things is None:
        all_things = globals()
    all_things[func_name] = _func

add_func_in_general_for_opt = add_func_in_general

def get_general_data_func(
        data_func, tokenizer=None, 
        max_epochs=100, ckpt_save_interval=50,
        col_name=None):
    def _func(key_params):
        if col_name is not None:
            key_params['col_name'] = col_name
        if tokenizer is None:
            _tokenizer = get_tokenizer_func()
        else:
            _tokenizer = tokenizer
        key_params['get_dataset_func'] = functools.partial(
                    data_func,
                    tokenizer=_tokenizer)
        key_params['max_epochs'] = max_epochs
        key_params['ckpt_save_interval'] = ckpt_save_interval
        return key_params
    return _func


def get_func_for_add_exp_seeds(
        DATA_KWARGS, KWARGS,
        ):
    def add_exp_seeds(
            exp_names, seeds, data_func, post_func,
            new_tokenizer=None,
            model_name=None):
        now_data_kwargs = copy.copy(DATA_KWARGS)
        if new_tokenizer is not None:
            now_data_kwargs['tokenizer'] = new_tokenizer

        now_kwargs = copy.copy(KWARGS)
        if model_name is not None:
            now_kwargs['model_name'] = model_name
        for exp_name, seed in zip(exp_names, seeds):
            add_func_in_general_for_opt(
                    func_name=exp_name,
                    data_func=get_general_data_func(
                        data_func,
                        **now_data_kwargs),
                    seed=seed, post_func=post_func,
                    **now_kwargs)
    return add_exp_seeds


def get_seq_funcs_to_update_key_params(funcs):
    def _func(key_params):
        for each_func in funcs:
            key_params = each_func(key_params)
        return key_params
    return _func


def add_set_epoch(key_params):
    key_params['need_set_epoch_hook'] = True
    key_params['persistent_workers'] = False
    return key_params


def change_max_epoch(
        key_params, new_max_epochs,
        specify_epoch=[2, 5, 10, 15, 20],
        save_interval=None):
    if save_interval is None:
        key_params['ckpt_save_interval'] = new_max_epochs // 4
    else:
        key_params['ckpt_save_interval'] = save_interval
    key_params['max_epochs'] = new_max_epochs
    key_params['specify_epoch'] = specify_epoch
    return key_params

def change_max_epoch_save_only_final(
        key_params, new_max_epochs):
    key_params['ckpt_save_interval'] = new_max_epochs
    key_params['max_epochs'] = new_max_epochs
    key_params['specify_epoch'] = [10,]
    return key_params


def add_set_epoch(key_params):
    key_params['need_set_epoch_hook'] = True
    key_params['persistent_workers'] = False
    return key_params

def get_name_seeds(exp_prfx):
    return dict(
            exp_names=[
                f'{exp_prfx}1', f'{exp_prfx}2', f'{exp_prfx}11', f'{exp_prfx}12',
                ],
            seeds=[1,2, 11,12,
                   ])


def add_tk_pad_collate_fn(key_params, tokenizer):
    if 'add_train_loader_kwargs' not in key_params:
        key_params['add_train_loader_kwargs'] = {}
    key_params['add_train_loader_kwargs'].update(
            {'collate_fn': functools.partial(
                tk_pad_collate_fn,
                tokenizer=tokenizer,
                )})
    return key_params


def get_func_for_replace_batch_processor_to_func(new_func):
    def _func(key_params):
        key_params['batch_processor_params'] = {
            'func': new_func,
            }
        return key_params
    return _func


def get_func_for_change_batch_size(new_batch_size):
    def _func(key_params):
        key_params['base_batch_size'] = new_batch_size
        key_params['desired_batch_size'] = new_batch_size
        return key_params
    return _func
