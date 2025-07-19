#!/bin/bash

# List of model names for the first set
first_set_model_names=("GPT2-small_tinydialogue_ordered_10n_1e-04" "GPT2-small_tinydialogue_reversed_10n_1e-04" "GPT2-small_tinydialogue_randomized_10n_1e-04")

# List of model names for the second set
second_set_model_names=("GPT2-small_tinydialogue_randomized_10n_no-speaker-labels_1e-04" "GPT2-small_tinydialogue_ABC_1-epoch_randomized_no-speaker-labels_1e-04" "GPT2-small_tinydialogue_ABC_20-epochs_randomized_no-speaker-labels_1e-04")

# Loop through each model name in the first set with Mom speaker label zorro formatting
for model_name in "${first_set_model_names[@]}"; do
    # Execute the Python command
    python babylm_eval_zorro.py "../tinydialogue/trained_GPT2_models/${model_name}" "decoder" "zorro_dialogue-format-tinydialogue_Mom" "../tinydialogue/Zorro_GPT2-small_tinydialogue_all_experiments_results.txt" "../tinydialogue/Zorro_GPT2-small_tinydialogue_all_experiments_final-avg-scores.csv"
    
    # Check if the Python script executed successfully
    if [ $? -eq 0 ]; then
        echo "Successfully processed ${model_name} with Mom format."
    else
        echo "Error processing ${model_name} with Mom format."
    fi
done


# Loop through each model name in the first set with Child speaker label zorro formatting
for model_name in "${first_set_model_names[@]}"; do
    # Execute the Python command
    python babylm_eval_zorro.py "../tinydialogue/trained_GPT2_models/${model_name}" "decoder" "zorro_dialogue-format-tinydialogue_Child" "../tinydialogue/Zorro_GPT2-small_tinydialogue_all_experiments_results.txt" "../tinydialogue/Zorro_GPT2-small_tinydialogue_all_experiments_final-avg-scores.csv"
    
    # Check if the Python script executed successfully
    if [ $? -eq 0 ]; then
        echo "Successfully processed ${model_name} with Child format."
    else
        echo "Error processing ${model_name} with Child format."
    fi
done

# Loop through each model name in the second set with original (no speaker label) zorro formatting
for model_name in "${second_set_model_names[@]}"; do
    # Execute the Python command
    python babylm_eval_zorro.py "../tinydialogue/trained_GPT2_models/${model_name}" "decoder" "zorro" "../tinydialogue/Zorro_GPT2-small_tinydialogue_all_experiments_results.txt" "../tinydialogue/Zorro_GPT2-small_tinydialogue_all_experiments_final-avg-scores.csv"
    
    # Check if the Python script executed successfully
    if [ $? -eq 0 ]; then
        echo "Successfully processed ${model_name} with original format."
    else
        echo "Error processing ${model_name} with original format."
    fi
done