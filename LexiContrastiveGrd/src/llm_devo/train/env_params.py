import os
from pt_framework.dist_utils import use_tpu
from ..env_vars import ROOT_DIR, ROOT_DIR_FREQ, ROOT_DIR_FREQ_ORG


MODEL_SAVE_FOLDER = os.environ.get(
        'MODEL_SAVE_FOLDER',
        os.path.join(
            ROOT_DIR_FREQ_ORG, 'models/'))
REC_SAVE_FOLDER = os.environ.get(
        'REC_SAVE_FOLDER',
        os.path.join(
            ROOT_DIR_FREQ_ORG, 'model_recs/'))
USE_TPU = use_tpu()
MANUAL_FORCE_TPU = int(os.environ.get('MANUAL_FORCE_TPU', 0)) == 1
USE_TPU = USE_TPU or MANUAL_FORCE_TPU
