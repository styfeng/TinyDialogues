#SOURCES: https://colab.research.google.com/github/huggingface/blog/blob/main/notebooks/01_how_to_train.ipynb#scrollTo=GlvP_A-THEEl and https://huggingface.co/blog/how-to-train

import sys
import torch
torch.cuda.is_available()
torch.cuda.empty_cache()
#torch_device = "cuda" if torch.cuda.is_available() else "cpu"

train_file = sys.argv[1]
val_file = sys.argv[2]
tokenizer_name = sys.argv[3]
output_folder = sys.argv[4]
voc_size = int(sys.argv[5])
text_chunk = int(sys.argv[6])
line_by_line = sys.argv[7]
lr = float(sys.argv[8])
num_epochs = int(sys.argv[9])
batch_size = int(sys.argv[10])
random_seed = int(sys.argv[11])

from transformers import RobertaConfig

config = RobertaConfig(
  architectures=[
    "RobertaForMaskedLM"
  ],
  attention_probs_dropout_prob=0.1,
  bos_token_id=0,
  eos_token_id=2,
  hidden_act="gelu",
  hidden_dropout_prob=0.1,
  hidden_size=768,
  initializer_range=0.02,
  intermediate_size=3072,
  layer_norm_eps=1e-05,
  max_position_embeddings=514,
  model_type="roberta",
  num_attention_heads=12,
  num_hidden_layers=12,
  pad_token_id=1,
  type_vocab_size=1,
  vocab_size=voc_size,
)


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)


from transformers import RobertaForMaskedLM

model = RobertaForMaskedLM(config=config)

model.num_parameters()



from datasets import load_dataset
from transformers import RobertaTokenizer, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader

# Load the plain text dataset, treating each line as a separate example
train_dataset = load_dataset('text', data_files={'train': train_file}, split='train')
val_dataset = load_dataset('text', data_files={'validation': val_file}, split='validation')

# Print a few raw examples to ensure order before tokenization
print("\n\nRAW TRAIN EXAMPLES BEFORE TOKENIZATION:\n")
for i in range(5):
    print(f"Example {i}: {train_dataset[i]['text']}\n")
print("\nnVAL EXAMPLE:\n")
print(val_dataset[0],'\n')


# Tokenize the dataset, preserving line-by-line behavior
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding="max_length", max_length=text_chunk)

# Define a function to concatenate text and chunk into sequences of max_length
def tokenize_and_chunk(examples):
    # Concatenate all the lines into one large text
    all_text = ' '.join(examples['text'])
    
    # Tokenize the concatenated text
    tokenized_text = tokenizer(all_text, truncation=False)['input_ids']
    
    # Chunk the tokenized text into blocks of 512 tokens (or text_chunk length)
    chunks = [tokenized_text[i:i+text_chunk] for i in range(0, len(tokenized_text), text_chunk)]
    
    # If the last chunk is smaller than text_chunk, pad it
    if len(chunks[-1]) < text_chunk:
        chunks[-1] += [tokenizer.pad_token_id] * (text_chunk - len(chunks[-1]))
    
    # Create attention masks (1 for real tokens, 0 for padding tokens)
    attention_masks = [[1] * len(chunk) + [0] * (text_chunk - len(chunk)) for chunk in chunks]
    #attention_masks = [[1] * min(len(chunk), text_chunk) + [0] * (text_chunk - len(chunk)) for chunk in chunks]

    # Return the chunks and the corresponding attention masks
    return {'input_ids': chunks, 'attention_mask': attention_masks}


if line_by_line.lower() == 'line_by_line':

    # Apply the tokenizer to the dataset
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)

    # Remove the 'text' column since it's no longer needed after tokenization
    train_dataset = train_dataset.remove_columns(['text'])
    val_dataset = val_dataset.remove_columns(['text'])

    # Set the format to PyTorch tensors
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    # Print a few examples after tokenization to check the order
    print("\n\nTOKENIZED TRAIN EXAMPLES (line_by_line):\n")
    for i in range(10):
        decoded_text = tokenizer.decode(train_dataset[i]['input_ids'], skip_special_tokens=False)
        print(f"Decoded Example {i}: {decoded_text}\n")
        
else:
    
    # Apply the tokenization and chunking function to the dataset
    train_dataset = train_dataset.map(tokenize_and_chunk, batched=True, batch_size=1000, remove_columns=['text'])
    val_dataset = val_dataset.map(tokenize_and_chunk, batched=True, batch_size=1000, remove_columns=['text'])

    # Set the format to return PyTorch tensors
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    # Print a few examples after tokenization and chunking to check the order
    print("\n\nTOKENIZED TRAIN EXAMPLES (chunking):\n")
    for i in range(10):
        # Convert the tensor to a list first, then flatten if needed
        chunk_list = train_dataset[i]['input_ids'].tolist()  # Convert tensor to list
        
        # Flatten if the list contains sublists
        if isinstance(chunk_list[0], list):
            flat_chunk = sum(chunk_list, [])
        else:
            flat_chunk = chunk_list  # If it's already flat, no need to flatten
        
        # Decode the flattened list
        decoded_chunk = tokenizer.decode(flat_chunk, skip_special_tokens=False)
        print(f"Decoded Chunk {i}: {decoded_chunk}\n")

# Create a data collator for masked language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)


from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir=output_folder,
    #overwrite_output_dir=True,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    do_train=True,
    do_eval=True,
    seed=random_seed,
    learning_rate=lr,
    load_best_model_at_end=True,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    save_total_limit=2,
    #prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)


import time
start = time.time()

trainer.train()
#trainer.train(resume_from_checkpoint=input_model)

end = time.time()
print("time taken (seconds): ", end-start)