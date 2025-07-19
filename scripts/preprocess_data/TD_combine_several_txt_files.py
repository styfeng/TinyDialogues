import sys

def combine_files(output_file, input_files):
    """Combine multiple text files into one output file."""
    
    with open(output_file, "w", encoding="utf-8") as out_f:
        for file in input_files:
            try:
                with open(file, "r", encoding="utf-8") as f:
                    out_f.writelines(f.readlines())  # Append contents of each file
                print(f"Added {file}")
            except FileNotFoundError:
                print(f"Warning: {file} not found. Skipping.")
    
    print(f"Combined text saved to {output_file}")

# Ensure correct usage
if len(sys.argv) < 3:
    print("Usage: python combine_txt_files.py <output_file.txt> <input_file1.txt> <input_file2.txt> ...")
    sys.exit(1)

# Get output filename and input filenames from command-line arguments
output_file = sys.argv[1]
input_files = sys.argv[2:]

# Run the function
combine_files(output_file, input_files)
