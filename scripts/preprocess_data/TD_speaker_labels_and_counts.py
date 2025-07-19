import re
from collections import Counter
import sys

file_path = sys.argv[1]

def get_speaker_statistics(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    # Regular expression to find all speaker labels surrounded by double asterisks
    speaker_pattern = r'\*\*(.*?)\*\*'
    speakers = re.findall(speaker_pattern, content)

    # Calculate frequency counts
    speaker_counts = Counter(speakers)

    # Total number of speaker labels for percentage calculation
    total_speakers = sum(speaker_counts.values())

    # Calculate percentages and sort by frequency in descending order
    speaker_stats = [
        (speaker, count, (count / total_speakers) * 100)
        for speaker, count in speaker_counts.items()
    ]
    speaker_stats.sort(key=lambda x: x[1], reverse=True)

    return speaker_stats

# Get speaker statistics
speaker_stats = get_speaker_statistics(file_path)

# Print the speaker statistics
print("Speaker Label | Count | Percentage")
print("-----------------------------------")
for speaker, count, percentage in speaker_stats:
    print(f"{speaker:14} | {count:5} | {percentage:7.2f}%")