Code for both ["Visual Grounding Helps Learn Word Meanings in Low-Data Regimes"](https://arxiv.org/abs/2310.13257) and ["Lexicon-Level Contrastive Visual-Grounding Improves Language Modeling"](https://arxiv.org/abs/2403.14551)

# Environment

Python 3.9, transformer package in huggingface, and datasets package in huggingface.

Training requires `pt_framework` repo ([link](https://github.com/chengxuz/pt_framework.git)).

Evaluation requires `lm_eval` repo ([link](https://github.com/chengxuz/lm-evaluation-harness.git)).

Install this repo via `pip install -e .`

# Model Training

## Dataset preparation

Conceptual-12M is used in both papers.
Because downloading the images of this dataset can be hard, we provide visual states precomputed from a DINO-pretrained ViT-Base.
These precomputed states and the other needed files are put in a zip file,
which can be downloaded from this [link](https://www.dropbox.com/scl/fi/xistjr06fwhg1ucelqgdk/Conceptual-12M.zip?rlkey=fyxl8j5yj3kxwgcqyn3mg6g0e&dl=0).

This repo uses environment variables to check where the dataset is and to decide where the models and evaluation results will be stored,
see the following two varialbes:
```
export ROOT_DIR_FREQ_ORG="/path/to/store_folder"
export DATASET_ROOT_DIR_FREQ="/path/to/dataset_folder"
```

The downloaded zip file should be extracted to `${DATASET_ROOT_DIR_FREQ}` so that the `Conceptual-12M` folder will be directly under `${DATASET_ROOT_DIR_FREQ}`.

## Ground-Only Training (Conceptual-12M)

To train the models, go to the `scripts` folder. The training command is generally the following:

```
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29123 general_train.py --setting ${SETTING}
```

The SETTING variable decides which model will be trained.

### LexiContrastive Grounding (LCG)

The LCG models use the following SETTING variable: `ground_only/exp_clip.py:idx_base_bs128_${size}_lcg_ly6_git_like_clip_s${seed}`.
Here the ${size} can be one of the following: `100K, 500K, 1M, 5M, 15M, 50M`, which represents the number of tokens in the training captions.
The ${seed} variable can be one of the following: `1, 2, 11, 12`.
The SETTING varialbe points the training script to a config function defined inside the `src/llm_devo/configs/ground_only/exp_clip.py` file.

### CLIP

The Visual + Language models in the first paper (also the CLIP models shown in the second paper) use the following SETTING variable:
`ground_only/exp_clip.py:idx_base_bs512_${size}_cached_git_like_np1_clip_s${seed}`.

The Visual + Word models in the first paper use the following SETTING variable:
`ground_only/exp_clip.py:idx_single_words_bs512_${size}_cached_git_like_np1_clip_s${seed}`.

### GIT

The Visual + Language models in the first paper (also the GIT models shown in the second paper) use the following SETTING variable:
`ground_only/exp_git_lang_only.py:idx_base_${size}_dino_cached_50M_gitl_s${seed}`.

The Visual + Word models in the first paper use the following SETTING variable:
`ground_only/exp_git_lang_only.py:idx_single_words_${size}_dino_cached_50M_gitl_s${seed}`.

### Language-Only

The Language-Only models in the both papers use the following SETTING variable:
`ground_only/exp_git_lang_only.py:txt_base_${size}_noimg_tie_lyrs_6_gitl_s${seed}`.

The Word-Only Baseline models in the first paper use the following SETTING variable:
`ground_only/exp_git_lang_only.py:txt_single_words_${size}_noimg_tie_lyrs_6_gitl_s${seed}`.

### Flamingo

COMING SOON.

## Mixed Unground-Ground Training

COMING SOON.

# Model Evaluation

## Word-Relatedness

See the README file in the `./src/llm_devo/word_sim/` folder.

## Semantic-Feature Prediction

See the README file in the `./src/llm_devo/word_norm/` folder.

## Lexical-Relation Prediction

See the README file in the `./src/llm_devo/lexical_relation/` folder.

## Part-Of-Speech Prediction

See the README file in the `./src/llm_devo/pos_pred/` folder.

# Context-based Word Understanding

See the README file in the `./src/llm_devo/word_understand/` folder.
