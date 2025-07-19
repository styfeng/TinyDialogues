import sys
import random
from tqdm import tqdm

# Get command-line arguments
input_file = sys.argv[1]
amount_labels = sys.argv[2].split(",")
train_amounts = list(map(int, sys.argv[3].split(",")))  # e.g., "100000000,50000000,20000000"
val_amounts = list(map(int, sys.argv[4].split(",")))  # e.g., "18000000,9000000,3500000"

# Check that the number of labels matches the number of train/val splits
if len(amount_labels) != len(train_amounts) or len(amount_labels) != len(val_amounts):
    print("Error: Number of amount labels must match the number of train/val split values.")
    sys.exit(1)

# Generate file names
output_base = input_file.replace(".txt", "")
output_files = [(f"{output_base}_{label}_train.txt", f"{output_base}_{label}_val.txt")
                for label in amount_labels]

def count_words(sentence):
    """Counts words in a given sentence."""
    return len(sentence.split())

def sample_dataset(conversations, train_limit, val_limit):
    """Samples train and validation sets based on word limits."""
    random.shuffle(conversations)  # Ensure randomness

    train_set = []
    val_set = []
    train_word_count = 0
    val_word_count = 0

    for conversation in tqdm(conversations):
        word_count = count_words(conversation)
        if train_word_count + word_count <= train_limit:
            train_set.append(conversation)
            train_word_count += word_count
        elif val_word_count + word_count <= val_limit:
            val_set.append(conversation)
            val_word_count += word_count

        if train_word_count >= train_limit and val_word_count >= val_limit:
            break

    return train_set, val_set

# Read input data
with open(input_file, "r", encoding="utf-8") as file:
    conversations = file.readlines()

for i, (train_limit, val_limit) in enumerate(zip(train_amounts, val_amounts)):
    # Sample train and val sets given word limits for each
    train_set, val_set = sample_dataset(conversations, train_limit, val_limit)

    # Write output files
    train_file, val_file = output_files[i]
    with open(train_file, "w", encoding="utf-8") as f_train:
        f_train.writelines(train_set)
    with open(val_file, "w", encoding="utf-8") as f_val:
        f_val.writelines(val_set)

    print(f"Train split ({train_limit} words) saved to {train_file}")
    print(f"Val split ({val_limit} words) saved to {val_file}")