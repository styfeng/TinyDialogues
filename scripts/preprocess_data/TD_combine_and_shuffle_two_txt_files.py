import sys
import random
random.seed(0)

def combine_and_shuffle(file1, file2, output_file):
    # Read lines from both files
    with open(file1, "r", encoding="utf-8") as f1, open(file2, "r", encoding="utf-8") as f2:
        lines = f1.readlines() + f2.readlines()

    # Shuffle the combined lines
    random.shuffle(lines)

    # Write shuffled lines to the output file
    with open(output_file, "w", encoding="utf-8") as out_f:
        out_f.writelines(lines)

    print(f"Shuffled output saved to {output_file}")

# Ensure correct usage
if len(sys.argv) != 4:
    print("Usage: python TD_combine_two_txt_files.py <file1.txt> <file2.txt> <output_file.txt>")
    sys.exit(1)

# Get filenames from command-line arguments
file1, file2, output_file = sys.argv[1], sys.argv[2], sys.argv[3]

# Run the function
combine_and_shuffle(file1, file2, output_file)
