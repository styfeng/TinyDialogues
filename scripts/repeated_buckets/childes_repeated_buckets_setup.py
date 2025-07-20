import sys

def main(input_file, output_file, num_buckets, repeats_per_bucket):
    # Read all lines from the input file
    with open(input_file, 'r') as file:
        lines = file.readlines()

    # Calculate the number of lines per bucket
    lines_per_bucket = len(lines) // num_buckets
    remainder = len(lines) % num_buckets

    # Prepare the output lines by repeating each bucket the specified number of times
    output_lines = []
    start_index = 0

    for i in range(num_buckets):
        # Add one extra line to the last buckets if there's a remainder
        end_index = start_index + lines_per_bucket + (1 if i < remainder else 0)
        bucket = lines[start_index:end_index]
        output_lines.extend(bucket * repeats_per_bucket)
        start_index = end_index

    # Write the output lines to the output file
    with open(output_file, 'w') as file:
        file.writelines(output_lines)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python childes_repeated_buckets_setup.py input.txt num_buckets repeats_per_bucket")
        sys.exit(1)

    input_path = sys.argv[1] #e.g., "CHILDES_train_ordered.txt", "CHILDES_train_reversed.txt", "CHILDES_train_randomized.txt". Files can be found in "data/CHILDES_data.zip"
    buckets = int(sys.argv[2])
    repeats = int(sys.argv[3])
    output_path = f"{input_path.split('/')[0]}/repeated_buckets/{input_path.split('/')[-1].replace('.txt',f'_{buckets}b_{repeats}n.txt')}"
    main(input_path, output_path, buckets, repeats)