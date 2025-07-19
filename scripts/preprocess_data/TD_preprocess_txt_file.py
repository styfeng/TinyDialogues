import sys
import re
import random
random.seed(0)


def filter_example(content):
    """Preprocess the content by filtering out invalid examples."""

    #if 'DIALOGUE:' not in content:
    #    return True
    
    if content.lower().count('dialogue:') > 1:
        return True
    
    if '-year-old' in content.lower():
        return True
    
    if '**' not in content:
        return True
    
    if len(content.split()) > 400:
        return True

    return False


def preprocess_content(content):
    """Preprocess the content by applying required transformations."""
    
    content = content.strip()
    
    # Remove "DIALOGUE:\n" at the beginning
    content = re.sub(r"^DIALOGUE:\n\n", "", content)
    content = re.sub(r"^DIALOGUE:\n", "", content)
    content = content.replace("**DIALOGUE:**", "")
    content = content.replace("DIALOGUE:", "")
    
    # Ensure newlines are explicit tokens and not treated as actual newlines
    content = content.replace("\n", "\\n")
    
    # Ensure spaces before and after double newlines (utterance separators)
    content = content.replace("\\n\\n", " \\n\\n ")

    # Get rid of extra newline tokens at beginning of lines
    if content.startswith("\\n"):
        content = content[2:]

    # Ensure all utterance separator tokens are double newline tokens with whitespace before and after
    content = content.replace("\\n'\\n**", ' \\n\\n **').replace('\\n**',' \\n\\n **')
    
    # Replace 'Toddler' speaker label with 'Child' speaker label
    content = content.replace("**Toddler**", "**Child**").replace("**toddler**", "**Child**").replace("**Toddler", "**Child").replace("Toddler**","Child**").replace("Toddler","Child").replace("toddler","child").replace("TODDLER","Child")
    
    # Replace all "curly/smart" quotes and apostraphes with straight ones:
    content = content.replace("’","'").replace("‘","'").replace("“","\"").replace("”","\"").strip()
    
    # Replace EOS token for GPT-2 training
    if '<|endoftext|>' not in content:
        content = content.strip() + ' <|endoftext|>'
    
    return content


def preprocess_txt_file(txt_file,out_file):
    
    bad_data_counter = 0
    kept_data_counter = 0
    kept_dialogues = []
    
    # Read lines from txt file
    with open(txt_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        
        # Apply filtering
        remove_example = filter_example(line)
        if remove_example:
            bad_data_counter += 1
            continue

        # Apply preprocessing
        cleaned_content = preprocess_content(line)
        
        #Further filter if "dialogue" still in content
        if 'dialogue' in cleaned_content.lower():
            bad_data_counter += 1
            continue
        
        kept_data_counter += 1
        kept_dialogues.append(cleaned_content)

    print(f"\n{kept_data_counter} examples kept") 
    print(f"{bad_data_counter} examples skipped")

    # Write cleaned dialogues to a .txt file
    with open(out_file, 'w') as file:
        for dialogue in kept_dialogues:
            file.write(dialogue + '\n')
    
    print("Preprocessed data written to .txt file")


# Get filenames from command-line arguments
input_file, output_file = sys.argv[1], sys.argv[2]

# Run the function
preprocess_txt_file(input_file, output_file)