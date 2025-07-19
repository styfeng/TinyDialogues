#!/bin/bash

MODEL_PATH=$1
TASK_NAME=$2
LR=${3:-5e-5}           # default: 5e-5
#PATIENCE=${4:-2}       # default: 10
BSZ=${4:-64}            # default: 64
MAX_EPOCHS=${5:-10}     # default: 10
SEED=${6:-12}           # default: 12

mkdir -p $MODEL_PATH/finetune/$TASK_NAME/

if [[ "$TASK_NAME" = "mnli" ]]; then
    VALID_NAME="validation_matched"
    OUT_DIR="mnli"
elif [[ "$TASK_NAME" = "mnli-mm" ]]; then
    VALID_NAME="validation_mismatched"
    TASK_NAME="mnli"
    OUT_DIR="mnli-mm"
else
    VALID_NAME="validation"
    OUT_DIR=$TASK_NAME
fi

if [[ "$TASK_NAME" = "mrpc" ]]; then
    BEST_METRIC="f1"
elif [[ "$TASK_NAME" = "qqp" ]]; then
    BEST_METRIC="f1"
else
    BEST_METRIC="accuracy"
fi

echo "TASK="$TASK_NAME
echo "BEST_METRIC="$BEST_METRIC

#CUDA_VISIBLE_DEVICES=`free-gpu` python finetune_classification.py \

python finetune_classification.py \
  --task_name_glue $TASK_NAME \
  --model_name_or_path $MODEL_PATH \
  --output_dir $MODEL_PATH/finetune/$OUT_DIR/ \
  --train_file filter-data/glue_filtered/$TASK_NAME.train.json \
  --validation_file filter-data/glue_filtered/$TASK_NAME.$VALID_NAME.json \
  --do_train \
  --do_eval \
  --use_fast_tokenizer False \
  --max_seq_length 128 \
  --per_device_train_batch_size $BSZ \
  --learning_rate $LR \
  --num_train_epochs $MAX_EPOCHS \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --overwrite_output_dir \
  --seed $SEED \
  --load_best_model_at_end True \
  --save_total_limit 1 \
  --metric_for_best_model $BEST_METRIC \
  --greater_is_better True