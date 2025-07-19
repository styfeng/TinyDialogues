import torch
import ipdb
import argparse
import transformers
import functools
import os
import re
import numpy as np
import pickle
from tqdm import tqdm


from lm_eval.base import BaseLM
from lm_eval.models import get_model, MODEL_REGISTRY
from lm_eval.models.gpt2 import HFLM
from llm_devo.env_vars import DEBUG

MODEL_REGISTRY.update({
        "facebook/opt-125m": HFLM,
        "facebook/opt-350m": HFLM,
        "facebook/opt-1.3b": HFLM,
        "facebook/opt-2.7b": HFLM,
        "facebook/opt-6.7b": HFLM,
        "facebook/opt-13b": HFLM,
        })


class LLMDevoModels(HFLM):
    def __init__(
            self, 
            model,
            tokenizer,
            device='',
            batch_size=16,
            extra_forward_mode=None,
            *args, **kwargs):
        BaseLM.__init__(self, *args, **kwargs)

        assert isinstance(device, str)
        assert isinstance(batch_size, int)

        if device:
            if device not in ["cuda", "cpu"]:
                device = int(device)
            self._device = torch.device(device)
            print(f"Using device '{device}'")
        else:
            print("Device not specified")
            print(f"Cuda Available? {torch.cuda.is_available()}")
            self._device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        self.gpt2 = model.to(self.device)
        self.gpt2.eval()

        # pretrained tokenizer for neo is broken for now so just hard-coding this to gpt2
        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.vocab_size

        # multithreading and batching
        self.batch_size_per_gpu = batch_size  # todo: adaptive batch size

        self.extra_forward_mode = extra_forward_mode
        if extra_forward_mode is not None:
            if extra_forward_mode.startswith('vis_w_img_'):
                self.image_mode = extra_forward_mode.split('_')[3]
                HFVisualLM.prepare_pixel_values(
                        self, processor_type="microsoft/git-large")
            elif extra_forward_mode == 'idx_black':
                valid_idxs = torch.from_numpy(np.asarray([-1])).to(self.device)
                self.extra_forward_kwargs = {
                        'valid_idxs': valid_idxs,
                        }
            elif extra_forward_mode == 'flava_black':
                from transformers import AutoProcessor
                processor = AutoProcessor.from_pretrained(
                        "facebook/flava-full")
                image = np.zeros((3, 256, 256))
                test_inputs = processor(
                        images=[image],
                        return_image_mask=True,
                        padding=True,
                        max_length=128,
                        return_tensors="pt",
                        )
                test_inputs['bool_masked_pos'] = torch.ones_like(test_inputs['bool_masked_pos'])
                self.extra_forward_kwargs = {
                        'bool_masked_pos': test_inputs['bool_masked_pos'].to(self.device),
                        'pixel_values': test_inputs['pixel_values'].to(self.device),
                        }
            else:
                raise NotImplementedError
        else:
            self.extra_forward_kwargs = {}

        # TODO: fix multi-gpu
        # gpus = torch.cuda.device_count()
        # if gpus > 1:
        #     self.gpt2 = nn.DataParallel(self.gpt2)

    def load_ckpt(self, ckpt_path):
        import pt_framework.checkpoint as checkpoint
        if torch.cuda.is_available():
            checkpoint.load_checkpoint(self.gpt2, ckpt_path)
        else:
            checkpoint.load_checkpoint(
                    self.gpt2, ckpt_path,
                    map_location='cpu')

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            ret_logits = self.gpt2(
                inps,
                **self.extra_forward_kwargs)[0][:, -inps.shape[1]:, :self.vocab_size]
            return ret_logits

    @property
    def max_length(self):
        try:
            max_len = self.gpt2.config.n_ctx
        except AttributeError:
            # gptneoconfig doesn't have n_ctx apparently
            max_len = getattr(
                    self.gpt2.config,
                    'max_position_embeddings',
                    128)
        if max_len < 0:
            max_len = 10000000
        return max_len

    @property
    def eot_token_id(self):
        if self.tokenizer.eos_token_id is not None:
            # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
            return self.tokenizer.eos_token_id
        else:
            # GIT does not have eos
            return self.tokenizer.sep_token_id


class AutoHFLM(HFLM):
    def __init__(
        self,
        device="cuda",
        pretrained="gpt2",
        revision="main",
        subfolder=None,
        tokenizer=None,
        batch_size=1,
        builder=None
    ):
        BaseLM.__init__(self)

        assert isinstance(device, str)
        assert isinstance(pretrained, str)
        assert isinstance(batch_size, int)

        if device:
            if device not in ["cuda", "cpu"]:
                device = int(device)
            self._device = torch.device(device)
            print(f"Using device '{device}'")
        else:
            print("Device not specified")
            print(f"Cuda Available? {torch.cuda.is_available()}")
            self._device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        # TODO: update this to be less of a hack once subfolder is fixed in HF
        if builder is None:
            self.gpt2 = transformers.AutoModel.from_pretrained(
                pretrained,
                revision=revision + ("/" + subfolder if subfolder is not None else ""),
            ).to(self.device)
        else:
            self.gpt2 = builder.from_pretrained(pretrained).to(self.device)
        self.gpt2.eval()

        # pretrained tokenizer for neo is broken for now so just hard-coding this to gpt2
        token_kwargs = {}
        if subfolder is not None:
            token_kwargs['subfolder'] = subfolder
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained if tokenizer is None else tokenizer,
            revision=revision,
            **token_kwargs)
        self.vocab_size = self.tokenizer.vocab_size
        if isinstance(
            self.tokenizer, (transformers.GPT2Tokenizer, transformers.GPT2TokenizerFast)
        ) and pretrained == 'gpt2':
            assert self.tokenizer.encode("hello\n\nhello") == [
                31373,
                198,
                198,
                31373,
            ], self.tokenizer.encode("hello\n\nhello")

        # multithreading and batching
        self.batch_size_per_gpu = batch_size  # todo: adaptive batch size


class HFVisualLM(HFLM):
    def __init__(
            self, 
            image_mode='black',
            pretrained="microsoft/git-large",
            *args, **kwargs):
        super().__init__(
                pretrained=pretrained,
                *args, **kwargs)
        self.image_mode = image_mode
        self.pretrained = pretrained
        self.prepare_pixel_values()

    def prepare_pixel_values(self, processor_type=None):
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(
                processor_type or self.pretrained)
        if self.image_mode == 'black':
            image = np.zeros((3, 256, 256))
        elif self.image_mode == 'white':
            image = np.ones((3, 256, 256)) * 255
        elif self.image_mode == 'gray':
            image = np.ones((3, 256, 256)) * 128
        else:
            raise NotImplementedError

        self.pixel_values = self.processor(
                images=image, return_tensors="pt").pixel_values.to(self.device)
        self.extra_forward_kwargs = {
                'pixel_values': self.pixel_values,
                }

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            return self.gpt2(
                    inps,
                    pixel_values=self.pixel_values,
                    )[0][:, :, :self.vocab_size]
