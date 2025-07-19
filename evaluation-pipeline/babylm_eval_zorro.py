import argparse
import lm_eval
import os
import json
import torch
import csv

TASKS = {
    "blimp": ["anaphor_agreement.json", "argument_structure.json", "binding.json",
              "determiner_noun_agreement.json", "ellipsis.json",
              "filler_gap.json", "irregular_forms.json", "island_effects.json",
              "npi_licensing.json", "quantifiers.json", "subject_verb_agreement.json",
              "case_subjective_pronoun.json","local_attractor"],
} #removed "control_raising.json" since not in zorro and added "case_subjective_pronoun.json","local_attractor" as they are in zorro but not blimp

CUDA_VISIBLE_DEVICES=0
print(torch.cuda.is_available())
device = 'cuda'


def accuracy_on_task(task_name, eval_model, template_name, num_fewshot):

    eval_task = lm_eval.get_task_list(task_name, template_names=[template_name])
    
    # Run evaluation and get results, including per-example results
    results = lm_eval.evaluate(model=eval_model, tasks=eval_task, seed=12, num_fewshot=num_fewshot)
    
    # Extract the overall accuracy and the per-example results
    accuracy = results['results'][0]['acc'] # Overall accuracy
    per_example_results = results['per_example_results']  # List of per-example results
    
    return accuracy, per_example_results


def check_and_create_final_avg_scores_csv(file_name):
    # Check if the file exists
    if not os.path.isfile(file_name):
        # If the file does not exist, create it with the default header
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write the header
            writer.writerow(["Model", "Format", "Final Average Score"])
        print(f"File '{file_name}' created with default header.")
    else:
        print(f"File '{file_name}' already exists.")


def append_row_to_final_avg_scores_csv(file_name, model_path, input_str, final_avg_score):
    # Format the final average score as a percentage with two decimal places
    final_avg_score_formatted = f"{final_avg_score:.2f}%"
    
    # Open the file in append mode
    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write the new row
        writer.writerow([model_path, input_str, final_avg_score_formatted])
    print(f"Appended row to '{file_name}'.")
 

def check_and_create_task_results_csv(file_name):
    # Check if the file exists
    if not os.path.isfile(file_name):
        # If the file does not exist, create it with the default header
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write the header
            writer.writerow(["Model", "Task", "Score"])
        print(f"File '{file_name}' created with default header.")
    else:
        print(f"File '{file_name}' already exists.")


def append_row_to_task_results_csv(file_name, model_path, task, task_score):
    # Format the final average score as a percentage with two decimal places
    #task_score_formatted = f"{final_avg_score:.2f}%"
    
    # Open the file in append mode
    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write the new row
        writer.writerow([model_path, task, task_score])
    print(f"Appended row to {file_name}: {task} with score {task_score}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str,
                        help="Path to huggingface model and tokenizer.")
    parser.add_argument("model_type", type=str, choices=["decoder only", "decoder", "encoder only", "encoder", "encoder-decoder",],
                        help="Language model architecture.")
    parser.add_argument("input_str", type=str)
    parser.add_argument("output_file", type=str)
    parser.add_argument("output_final_avg_score_csv", type=str)
    parser.add_argument("output_task_results_csv", type=str)
    parser.add_argument("output_all_results_json", type=str)
    parser.add_argument("--tasks", "-t", type=str, choices=["blimp", "glue"], default="blimp",
                        help="Tasks on which we evaluate.")
    parser.add_argument("--num_fewshot", "-n", type=int, default=0,
                        help="Number of few-shot examples to show the model for each test example.")
    parser.add_argument("--trust_remote_code", "-r", action="store_true",
                        help="Trust remote code (e.g. from huggingface) when loading model.")
    args = parser.parse_args()

    MODEL_TYPE_REMAP = {"decoder only": "hf-causal", "decoder": "hf-causal",
                        "encoder only": "hf-mlm", "encoder": "hf-mlm",
                        "encoder-decoder": "hf-seq2seq",}
    eval_model = lm_eval.get_model(MODEL_TYPE_REMAP[args.model_type],
                                   pretrained=args.model_path,
                                   trust_remote_code=args.trust_remote_code,
                                   device="cuda")
    print(f'Type of evaluation: {args.input_str}')
    print(f'Output file to write results to: {args.output_file}')

    if not os.path.isfile(args.output_file):
        with open(args.output_file, 'w') as file:
            file.write('')
        print(f"File '{args.output_file}' created.")
    else:
        print(f"File '{args.output_file}' already exists.")

    check_and_create_final_avg_scores_csv(args.output_final_avg_score_csv)
    check_and_create_task_results_csv(args.output_task_results_csv)
    
    tasks = []
    if args.tasks == "all":
        for task_type in TASKS.keys():
            tasks.extend(TASKS[task_type])
    else:
        tasks = TASKS[args.tasks]

    accuracies = {}
    all_per_example_results = []
    
    # Iterate through tasks, get accuracies and per-example results
    for task in tasks:
        if task in TASKS["blimp"]:
            template = "null_prompt"
            task_title = task.split(".json")[0]
            if args.input_str != 'original':
                task = f"blimp_from_file:filter-data_{args.input_str}/{task}"
        else:
            raise ValueError("Unrecognized task!")
        
        accuracy, per_example_results = accuracy_on_task(task, eval_model, template, args.num_fewshot)
        accuracies[task_title] = accuracy
        all_per_example_results.extend(per_example_results)
        filtered_task_results = [{key: result[key] for key in ["pred", "target"] if key in result} for result in per_example_results]
        
        print(f"{task_title}:\t{accuracies[task_title] * 100:.2f}%")
        
        # Write per-task and per-example results to file
        out_path = os.path.join(args.model_path, f"zeroshot_{args.input_str}", task_title, "eval_results.json")
        out_dir = os.path.dirname(out_path)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        with open(out_path, 'w') as out_file:
            json.dump({"eval_accuracy": accuracies[task_title], "per_example_results": filtered_task_results}, out_file)

    # Save all per-example results for statistical tests
    per_example_results_file = args.output_all_results_json # Use the JSON Lines format (.jsonl)
    with open(per_example_results_file, 'a') as file:
        # Filter each dictionary in the list to only include "pred" and "target"
        filtered_results = [{key: result[key] for key in ["pred", "target"] if key in result} for result in all_per_example_results]
        # Write the entire list of filtered results as a single line in the file
        file.write(json.dumps(filtered_results) + "\n")

    # Print scores
    print("\nScores:")
    total_score = 0
    score_count = 0

    with open(args.output_file, 'a') as file:
        file.write(f'Zorro Scores for: {args.model_path} | {args.input_str}\n\n')
        for task in accuracies.keys():
            print(f"{task}:\t{accuracies[task] * 100:.2f}%")
            file.write(f"{task}:\t{accuracies[task] * 100:.2f}%\n")
            append_row_to_task_results_csv(args.output_task_results_csv, args.model_path, task, accuracies[task])
            score_count += 1
            total_score += accuracies[task] * 100
        final_avg_score = total_score / score_count
        print(f"FINAL AVERAGE SCORE: {final_avg_score:.2f}%")
        file.write(f"\nFINAL AVERAGE SCORE: {final_avg_score:.2f}%\n\n\n\n")
    
    #final_avg_score = total_score / score_count
    append_row_to_final_avg_scores_csv(args.output_final_avg_score_csv, args.model_path, args.input_str, final_avg_score)