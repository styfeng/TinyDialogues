#!/bin/bash

MODEL_PATH=$1
LR=${2:-5e-5}
BSZ=${3:-64}
MAX_EPOCHS=${4:-10}
SEED=${5:-12}

# use default hyperparameters
for task in {"cola","sst2","mrpc","qqp","mnli","mnli-mm","qnli","rte","boolq","multirc","wsc"}; do
    # qsub -q g.q -cwd -j y -l hostname="b1[123456789]|c0*|c1[13456789],ram_free=10G,mem_free=10G,gpu=1" finetune_model.sh $MODEL_PATH $task
	./finetune_model.sh $MODEL_PATH $task $LR $BSZ $MAX_EPOCHS $SEED
done