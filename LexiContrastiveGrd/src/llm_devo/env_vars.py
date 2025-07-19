import os
import sys
from os.path import expanduser

home = expanduser("~")
ROOT_DIR = os.environ.get(
        'ROOT_DIR',
        home)
ROOT_DIR_FREQ = os.environ.get(
        'ROOT_DIR_FREQ',
        ROOT_DIR)
ROOT_DIR_FREQ_ORG = os.environ.get(
        'ROOT_DIR_FREQ_ORG',
        os.path.join(ROOT_DIR_FREQ, 'llm_devo'))
DATASET_ROOT_DIR = os.environ.get(
        'DATASET_ROOT_DIR',
        os.path.join(ROOT_DIR, 'llm_datasets'))
DATASET_ROOT_DIR_FREQ = os.environ.get(
        'DATASET_ROOT_DIR_FREQ',
        DATASET_ROOT_DIR)
DEBUG = int(os.environ.get(
        'DEBUG',
        '0')) == 1

def get_text_eval():
    TEXT_EVAL = int(os.environ.get(
            'TEXT_EVAL',
            '0')) == 1
    return TEXT_EVAL

def enable_text_eval():
    os.environ['TEXT_EVAL'] = '1'
