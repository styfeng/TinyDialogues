import pickle
import pdb
import os
import json
from itertools import product
import sys
import re
import copy
import numpy as np
import csv
import functools
from tqdm import tqdm
import multiprocessing


def naive_expand(
        added_keys,
        content_rep,
        default_added_keys=[
            'epoch_2.pth',
            'epoch_5.pth',
            'epoch_10.pth',
            'epoch_15.pth',
            'epoch_20.pth',
            'epoch_30.pth',
            'epoch_60.pth',
            ]):
    new_data = {}
    for key in added_keys + default_added_keys:
        new_data[key] = content_rep
    return new_data

def general_load_raw_results(pkl_path):
    raw_data = pickle.load(open(pkl_path, 'rb'))
    if os.path.basename(os.path.dirname(pkl_path)) == 'pretrained':
        raw_data = naive_expand(
                ['pretrained',],
                raw_data)
    if os.path.basename(
            os.path.dirname(os.path.dirname(
                pkl_path))) == 'untrained':
        raw_data = naive_expand(
                ['untrained',],
                raw_data)
    return raw_data
