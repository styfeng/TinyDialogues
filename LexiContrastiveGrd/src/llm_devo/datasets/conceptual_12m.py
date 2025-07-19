import torch
from PIL import Image
from torch.utils.data import Dataset
import os
import numpy as np
import pickle
from tqdm import tqdm
import inspect
import datetime
import ipdb
from itertools import product

from transformers import AutoProcessor
from ..env_vars import DATASET_ROOT_DIR_FREQ


class Conceptual12M(Dataset):
    def __init__(
            self, image_processor_func,
            root_dir=os.path.join(DATASET_ROOT_DIR_FREQ, 'Conceptual-12M/jpgs'), 
            text_meta_path=os.path.join(DATASET_ROOT_DIR_FREQ, 'Conceptual-12M/cc12m_texts.txt'),
            idx_segs=['1d5m'],
            text_only=True, verbose=False,
            tokenizer=None,
            sample_rel_num=None,
            sample_seed='No',
            use_trivial_image=None,
            drop_image=None,
            add_image_prefix_to_txt=False,
            all_valid_idxs=None,
            idx_remap_file=None,
            new_text_meta_path=None,
            simple_new_text_path=None,
            repeat_dataset=None,
            sentence_list_expand=None,
            return_valid_idx=False,
            random_shuffle_simple_sentences=None,
            use_cached_visual_embd=None,
            ):
        self.image_processor = image_processor_func()
        self.root_dir = root_dir
        self.idx_segs = idx_segs
        self.text_meta_path = text_meta_path
        self.text_only = text_only
        self.verbose = verbose
        self.tokenizer = tokenizer
        self.sample_rel_num = sample_rel_num
        self.sample_seed = sample_seed
        self.use_trivial_image = use_trivial_image
        self.drop_image = drop_image
        self.add_image_prefix_to_txt = add_image_prefix_to_txt
        self.all_valid_idxs = all_valid_idxs
        self.repeat_dataset = repeat_dataset
        self.sentence_list_expand = sentence_list_expand
        self.return_valid_idx = return_valid_idx
        self.use_cached_visual_embd = use_cached_visual_embd
        if self.use_cached_visual_embd is not None:
            self.load_cached_visual_embd()
        if idx_remap_file is not None:
            assert new_text_meta_path is not None
            self.idx_remap = np.load(idx_remap_file)
            with open(new_text_meta_path, 'r') as fin:
                self.new_text_metas = fin.readlines()
        else:
            self.idx_remap = None
        if self.drop_image is not None:
            assert self.use_trivial_image is not None
        if self.use_trivial_image is not None:
            self.prepare_trivial_image()

        with open(self.text_meta_path, 'r') as fin:
            self.text_metas = fin.readlines()

        self.load_valid_idxs()
        if simple_new_text_path is not None:
            if simple_new_text_path.endswith('pkl'):
                self.simple_new_sentences = pickle.load(
                        open(simple_new_text_path, 'rb'))
            else:
                raise NotImplementedError
            assert len(self.simple_new_sentences) >= len(self.all_valid_idxs)
            if random_shuffle_simple_sentences is not None:
                np.random.seed(random_shuffle_simple_sentences)
                self.simple_new_sentences = list(np.random.permutation(
                        self.simple_new_sentences))
        else:
            self.simple_new_sentences = None

        self.expand_sentence_list_if_needed()
        self.repeat_dataset_if_needed()

    def load_cached_visual_embd(self):
        cached_hidden_states_path = self.use_cached_visual_embd
        tmp_contents = pickle.load(
                open(cached_hidden_states_path, 'rb'))
        self.cached_vis_hidden_states = torch.from_numpy(
                np.asarray(tmp_contents['states']))
        if self.cached_vis_hidden_states.ndim == 2:
            self.cached_vis_hidden_states\
                    = self.cached_vis_hidden_states.unsqueeze(dim=1)

    def expand_sentence_list_if_needed(self):
        if self.sentence_list_expand is not None:
            assert self.simple_new_sentences is not None
            self.simple_new_sentences = self.simple_new_sentences[:len(self.all_valid_idxs)]
            if self.sentence_list_expand == 'single_word_expand':
                new_valid_idxs = []
                new_sentences = []
                for _idx, word_list in zip(
                        self.all_valid_idxs,
                        self.simple_new_sentences):
                    if len(word_list) == 0:
                        word_list = ['']
                    for word in word_list:
                        new_valid_idxs.append(_idx)
                        new_sentences.append(' ' + word)
                self.all_valid_idxs = new_valid_idxs
                self.simple_new_sentences = new_sentences
            else:
                raise NotImplementedError

    def repeat_dataset_if_needed(self):
        if self.repeat_dataset is not None:
            if self.simple_new_sentences is not None:
                self.simple_new_sentences = self.simple_new_sentences[:len(self.all_valid_idxs)]
                self.simple_new_sentences = np.repeat(
                        self.simple_new_sentences, self.repeat_dataset)
            self.all_valid_idxs = np.repeat(
                    self.all_valid_idxs,
                    self.repeat_dataset)

    def prepare_trivial_image(self):
        if self.use_trivial_image == 'white':
            image = np.ones((3, 256, 256)) * 255
        elif self.use_trivial_image == 'black':
            image = np.zeros((3, 256, 256))
        elif self.use_trivial_image == 'special_average_2of3':
            import llm_devo.models.ave_vis_states_for_words as ave_vis_states_for_words
            special_embd_path = os.path.join(
                    ave_vis_states_for_words.RESULT_DIR,
                    'dino',
                    'ave_lex_norm_sim_conceptual_12m.get_CC12M_50M_base.pkl',
                    )
            special_embds = pickle.load(open(special_embd_path, 'rb'))
            # words and embds in a dict
            self.special_average_word_map = {
                    _word: _idx
                    for _idx, _word in enumerate(
                        special_embds['words'],
                        )}
            return
        else:
            raise NotImplementedError
        self.trivial_image = self.image_processor(
                images=[image], return_tensors="pt")['pixel_values'][0]

    def load_valid_idxs(self):
        if self.all_valid_idxs is None:
            all_valid_idxs = []
            for idx_seg in self.idx_segs:
                valid_idx_path = os.path.join(
                        DATASET_ROOT_DIR_FREQ,
                        'Conceptual-12M',
                        'valid_idxs',
                        f'st{idx_seg}.pkl')
                all_valid_idxs.extend(
                        pickle.load(
                            open(valid_idx_path, 'rb')))
            self.all_valid_idxs = all_valid_idxs

        if self.sample_rel_num is not None:
            if self.sample_seed == 'No':
                subselect_idx = np.arange(
                        int(len(self.all_valid_idxs) * self.sample_rel_num))
            else:
                raise NotImplementedError
            self.all_valid_idxs = [
                    self.all_valid_idxs[_idx]
                    for _idx in subselect_idx]

    def __len__(self):
        return len(self.all_valid_idxs)

    def get_img_path(self, curr_idx):
        img_path = os.path.join(
            self.root_dir, f'img_{curr_idx:08}.jpg')
        return img_path

    def get_raw_text(self, curr_idx):
        if self.simple_new_sentences is not None:
            return self.simple_new_sentences[self.now_idx]
        if self.idx_remap is None:
            raw_txt = self.text_metas[curr_idx]
        else:
            new_idx = int(self.idx_remap[curr_idx])
            if new_idx == -1:
                raw_txt = self.text_metas[curr_idx]
            else:
                raw_txt = self.new_text_metas[new_idx]
        return raw_txt[:-1]

    def get_texts(self, curr_idx):
        raw_txt = self.get_raw_text(curr_idx)
        if self.text_only:
            texts = raw_txt
        else:
            if self.add_image_prefix_to_txt == False:
                texts = raw_txt
            elif self.add_image_prefix_to_txt == True:
                texts = '<image> ' + raw_txt
            elif isinstance(self.add_image_prefix_to_txt, float):
                if self.random_drop(1 - self.add_image_prefix_to_txt):
                    texts = raw_txt
                else:
                    texts = '<image> ' + raw_txt
            else:
                raise NotImplementedError
        return texts

    def sample_idx(self):
        return np.random.randint(
                low=self.idx_min, high=self.idx_max)

    def random_drop(self, prob=None):
        now_sample = np.random.uniform()
        if prob is None:
            prob = self.drop_image
        if now_sample < prob:
            return True
        else:
            return False

    def __getitem__(self, idx):
        self.now_idx = idx
        curr_idx = self.all_valid_idxs[idx]
        return_data = {}
        texts = self.get_texts(curr_idx)
        return_data['text'] = texts

        if self.return_valid_idx:
            return_data['valid_idxs'] = torch.tensor(curr_idx)
            if self.use_trivial_image is not None:
                if (self.drop_image is None) or (self.random_drop()):
                    if self.use_trivial_image == 'black':
                        return_data['valid_idxs'] = torch.tensor(-1)
                    elif self.use_trivial_image == 'special_average_2of3':
                        # leave the -1 to -99 for trivial images
                        now_word = str(texts).split(' ')[-2]
                        if now_word in self.special_average_word_map:
                            return_data['valid_idxs'] = torch.tensor(
                                    -100-self.special_average_word_map[now_word])

        if self.use_cached_visual_embd is not None:
            #return_data['visual_embds'] = self.cached_vis_hidden_states[curr_idx]
            return_data['visual_embds'] = self.cached_vis_hidden_states[idx]

        if self.text_only:
            if self.verbose:
                print(self.text_metas[curr_idx][:-1])
            return return_data
        if self.use_trivial_image is not None:
            if (self.drop_image is None) or (self.random_drop()):
                return_data['pixel_values'] = self.trivial_image
                return return_data

        img_path = self.get_img_path(curr_idx)
        try:
            image = Image.open(img_path)
            image = self.image_processor(
                    images=[image], return_tensors="pt")['pixel_values']
        except:
            #print(img_path)
            image = Image.fromarray(
                    np.ones((3, 256, 256)) * 255,
                    'RGB')
            image = self.image_processor(
                    images=[image], return_tensors="pt")['pixel_values']
        return_data['pixel_values'] = image[0]
        return return_data

    def count_num_of_tks(self, cut_len=128):
        assert self.tokenizer is not None
        num_of_tks = 0
        for line in tqdm(self):
            txt_in_tks = self.tokenizer(line['text'])
            num_of_tks += min(cut_len, len(txt_in_tks.input_ids))
        return num_of_tks


def get_CC12M_idx0m_base(
        tokenizer=None,
        just_dataset='self',
        processor_func=None,
        *args, **kwargs):
    assert just_dataset in ['self', True]
    if processor_func is None:
        model_name = "microsoft/git-large"
        processor_func = lambda: AutoProcessor.from_pretrained(model_name)
    dataset = Conceptual12M(
            image_processor_func=processor_func,
            tokenizer=tokenizer,
            idx_segs=['0m'],
            *args, **kwargs)
    return dataset


SCALE_TO_SAMPLE_REL_NUM = {
        '100K': 0.52 / 150,
        '500K': 0.52 / 30,
        '1M': 0.52 / 15,
        '2d5M': 0.52 / 6,
        '5M': 0.52 / 3,
        '7d5M': 0.52 / 2,
        '10M': 0.52 / 3 * 2,
        '15M': 0.52,
        '25M': 0.87 * 0.5,
        '30M': 0.87 * 0.6,
        '50M': 0.87,
        '100M': 0.87,
        '150M': 0.87,
        }
SCALE_TO_REPEAT_DATASET = {
        '100K': 10,
        '500K': 2,
        }
SCALE_TO_IDX_SEG = {
        '25M': ['1d5m', '3m'],
        '30M': ['1d5m', '3m'],
        '50M': ['1d5m', '3m'],
        '100M': ['1d5m', '3m', '4d5m', '6m'],
        '150M': ['1d5m', '3m', '4d5m', '6m', '7d5m', '9m'],
        }
SIZE_FOR_EXPS = [
        '100K', '500K', '1M', 
        '2d5M', '5M', '10M',
        '7d5M', '15M', '25M', '30M',
        '50M', '100M', '150M']
def add_base_funcs_for_scale(
        now_scale='15M',
        ):
    added_kwargs = dict(
            sample_rel_num=SCALE_TO_SAMPLE_REL_NUM[now_scale],
            repeat_dataset=SCALE_TO_REPEAT_DATASET.get(
                now_scale, None),
            idx_segs=SCALE_TO_IDX_SEG.get(
                now_scale, ['1d5m']),
            )
    base_func_name = f'get_CC12M_{now_scale}_base'
    def _base_func(
            tokenizer=None,
            just_dataset='self',
            processor_func=None,
            *args, **kwargs):
        for key in added_kwargs:
            if key in kwargs:
                print(f'{base_func_name} already have {key} defined!')
                return
        kwargs.update(added_kwargs)
        assert just_dataset in ['self', True]
        if processor_func is None:
            model_name = "microsoft/git-large"
            processor_func = lambda: AutoProcessor.from_pretrained(model_name)
        dataset = Conceptual12M(
                image_processor_func=processor_func,
                tokenizer=tokenizer,
                text_only=False,
                *args, **kwargs)
        return dataset

    txt_base_func_name = f'get_CC12M_{now_scale}_txt_base'
    def _txt_base_func(
            tokenizer=None,
            just_dataset='self',
            processor_func=None,
            *args, **kwargs):
        for key in added_kwargs:
            if key in kwargs:
                print(f'{txt_base_func_name} already have {key} defined!')
                return
        kwargs.update(added_kwargs)
        assert just_dataset in ['self', True]
        if processor_func is None:
            model_name = "microsoft/git-large"
            processor_func = lambda: AutoProcessor.from_pretrained(model_name)
        dataset = Conceptual12M(
                image_processor_func=processor_func,
                tokenizer=tokenizer,
                text_only=True,
                *args, **kwargs)
        return dataset

    flex_base_func_name = f'get_CC12M_{now_scale}_flex_base'
    def _flex_base_func(
            tokenizer=None,
            just_dataset='self',
            processor_func=None,
            *args, **kwargs):
        for key in added_kwargs:
            if key in kwargs:
                print(f'{flex_base_func_name} already have {key} defined!')
                return
        kwargs.update(added_kwargs)
        assert just_dataset in ['self', True]
        if processor_func is None:
            model_name = "microsoft/git-large"
            processor_func = lambda: AutoProcessor.from_pretrained(model_name)
        dataset = Conceptual12M(
                image_processor_func=processor_func,
                tokenizer=tokenizer,
                *args, **kwargs)
        return dataset

    all_things = globals()
    all_things[base_func_name] = _base_func
    all_things[txt_base_func_name] = _txt_base_func
    all_things[flex_base_func_name] = _flex_base_func

for size in SIZE_FOR_EXPS:
    add_base_funcs_for_scale(now_scale=size)


SIMPLE_RW_NAME_TO_TASK = {
        }
NAME_TO_LIST_EXPAND = {
        'nctx_words': 'single_word_expand',
        'single_words': 'single_word_expand',
        }
def add_mutli_scale_txt_simple_rw_funcs(
        simple_rw_name,
        now_scale='15M',
        ):
    scale_in_path = '50M'
    if now_scale in ['150M']:
        scale_in_path = now_scale
    simple_rw_path = os.path.join(
        DATASET_ROOT_DIR_FREQ,
        'Conceptual-12M/captions_rewriten_simple',
        SIMPLE_RW_NAME_TO_TASK.get(simple_rw_name, simple_rw_name),
        f'conceptual_12m.get_CC12M_{scale_in_path}_txt_base.pkl',
        )
    def _func(
            tokenizer=None,
            just_dataset='self',
            processor_func=None,
            *args, **kwargs):
        assert just_dataset in ['self', True]
        if processor_func is None:
            model_name = "microsoft/git-large"
            processor_func = lambda: AutoProcessor.from_pretrained(model_name)
        dataset = Conceptual12M(
                image_processor_func=processor_func,
                tokenizer=tokenizer,
                sample_rel_num=SCALE_TO_SAMPLE_REL_NUM[now_scale],
                simple_new_text_path=simple_rw_path,
                repeat_dataset=SCALE_TO_REPEAT_DATASET.get(
                    now_scale, None),
                sentence_list_expand=NAME_TO_LIST_EXPAND.get(
                    simple_rw_name, None),
                idx_segs=SCALE_TO_IDX_SEG.get(
                    now_scale, ['1d5m']),
                *args, **kwargs)
        return dataset
    now_name = f'get_CC12M_{now_scale}_{simple_rw_name}'
    all_things = globals()
    all_things[now_name] = _func


SIMPLE_RW_KEYS = [
         'nctx_words',
         'single_words',
         ]
for now_scale, simple_rw_name in product(
        ['100K', '500K', '1M', '5M', '15M', '50M', '150M'],
        SIMPLE_RW_KEYS):
    add_mutli_scale_txt_simple_rw_funcs(
            simple_rw_name=simple_rw_name,
            now_scale=now_scale)
