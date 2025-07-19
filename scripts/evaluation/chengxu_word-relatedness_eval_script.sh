#!/bin/bash

# List of model names for the first set
first_set_model_names=("GPT2-small_tinydialogue_ordered_10n_1e-04" "GPT2-small_tinydialogue_reversed_10n_1e-04" "GPT2-small_tinydialogue_randomized_10n_1e-04")

# List of model names for the second set
second_set_model_names=('Buckets/GPT2-small_CHILDES_ordered_5b_10n_1e-04','Buckets/Randomized/GPT2-small_CHILDES_randomized_5b_10n_1e-04','Buckets/Reversed/GPT2-small_CHILDES_reversed_5b_10n_1e-04')

# Loop through each model name in the first set
for model_name in "${first_set_model_names[@]}"; do
    # Execute the Python command
    python eval_word_sim.py --ckpt_path "/mnt/d/babyLM_project/tinydialogue/trained_GPT2_models/${model_name}" --output_file "/mnt/d/babyLM_project/tinydialogue/chengxu_word-relatedness_GPT2-small_tinydialogue_all_experiments_results.txt" --output_csv "/mnt/d/babyLM_project/tinydialogue/chengxu_word-relatedness_GPT2-small_tinydialogue_all_experiments_best-layer-scores.csv"
    
    # Check if the Python script executed successfully
    if [ $? -eq 0 ]; then
        echo "Successfully processed ${model_name}."
    else
        echo "Error processing ${model_name}."
    fi
done

# Loop through each model name in the second set
for model_name in "${second_set_model_names[@]}"; do
    # Execute the Python command
    python eval_word_sim.py --ckpt_path "/mnt/d/babyLM_project/CHILDES/trained_GPT2_models/${model_name}" --output_file "/mnt/d/babyLM_project/CHILDES/chengxu_word-relatedness_GPT2-small_CHILDES_all_experiments_results.txt" --output_csv "/mnt/d/babyLM_project/CHILDES/chengxu_word-relatedness_GPT2-small_CHILDES_all_experiments_best-layer-scores.csv"
    
    # Check if the Python script executed successfully
    if [ $? -eq 0 ]; then
        echo "Successfully processed ${model_name}."
    else
        echo "Error processing ${model_name}."
    fi
done