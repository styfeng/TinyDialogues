import pandas as pd
import os
import pickle
from llm_devo.env_vars import ROOT_DIR, ROOT_DIR_FREQ_ORG
import llm_devo


def load_aoa_data():
    aoa_path = os.path.join(
            llm_devo.__path__[0],
            'utils/aoa/AoA_ratings_Kuperman_et_al_BRM.xlsx')
    data = pd.read_excel(aoa_path)
    words = list(data['Word'])
    aoas = list(data['Rating.Mean'])
    aoa_dict = dict()
    for word, aoa in zip(words, aoas):
        aoa_dict[word] = aoa
    return aoa_dict
