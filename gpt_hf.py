import argparse
import torch
import os
from torch.utils.data import Dataset
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling
)

# ------------
# Argument Parser
# ------------
parser = argparse.ArgumentParser(description='Fine-tune Polish GPT-2 (HuggingFace)')

# I/O paths
parser.add_argument('--input', type=str, default='assets/input/input2.txt', help='Path to input text file')
parser.add_argument('--output_dir', type=str, default='assets/hf_model', help='Directory to save the fine-tuned model')
parser.add_argument('--model_name', type=str, default='flax-community/papuGaPT2', help='Base model from HF Hub')

# Hyperparameters
parser.add_argument('--batch_size', type=int, default=4, help='Batch size per device')
parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
parser.add_argument('--block_size', type=int, default=256, help='Context length (max 1024 for GPT-2)')
parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Gradient accumulation steps')
parser.add_argument('--prompt', type=str, default='', help='Text prompt to start generation. Leave empty for unconditional generation.')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
print(f"Loading base model: {args.model_name}")

# ------------
# 1. Load Model & Tokenizer
# ------------
# We use the Auto classes or specific GPT2 classes. PapuGaPT2 is GPT2 architecture.
tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
model = GPT2LMHeadModel.from_pretrained(args.model_name)

# GPT-2 usually doesn't have a pad token, we set it to eos_token for batching
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ------------
# 2. Prepare Dataset
# ------------
class TextFileDataset(Dataset):
    def __init__(self, txt_path, tokenizer, block_size):
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"Input file not found: {txt_path}")
            
        print(f"Reading data from {txt_path}...")
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        print("Tokenizing data...")
        # Add EOS token at the end of the text to help model learn termination
        text = text + tokenizer.eos_token
        self.tokens = tokenizer.encode(text)
        self.block_size = block_size
        print(f"Total tokens: {len(self.tokens)}")

    def __len__(self):
        # Drop the last incomplete chunk
        return len(self.tokens) // self.block_size

    def __getitem__(self, i):
        # We grab a chunk of size 'block_size'
        start_idx = i * self.block_size
        end_idx = start_idx + self.block_size
        
        # Transformers Trainer expects input_ids. 
        # DataCollatorForLanguageModeling will handle creating 'labels' automatically (shifting)
        return {
            "input_ids": torch.tensor(self.tokens[start_idx:end_idx], dtype=torch.long)
        }

train_dataset = TextFileDataset(args.input, tokenizer, args.block_size)

# ------------
# 3. Setup Trainer
# ------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False, # We are doing Causal Language Modeling (CLM), not Masked LM
)

training_args = TrainingArguments(
    output_dir=args.output_dir,
    # overwrite_output_dir=True,  # Removed to fix TypeError in newer transformers versions
    num_train_epochs=50, # ZWIĘKSZONE DRASTYCZNIE, aby model "zapomniał" prozę i nauczył się struktury wiersza
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps, # Simulate larger batch size
    learning_rate=args.learning_rate,
    weight_decay=0.1, # ZWIĘKSZONE, aby wymusić prostszą strukturę wag (rzadszą)
    logging_steps=10,
    save_steps=500,
    save_total_limit=3,
    prediction_loss_only=True,
    fp16=torch.cuda.is_available(), # Use mixed precision if on GPU
    dataloader_num_workers=0, # Windows often struggles with multiprocessing in dataloaders
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

# ------------
# 4. Train
# ------------
print("Starting training...")
trainer.train()

print(f"Saving model to {args.output_dir}...")
trainer.save_model()
tokenizer.save_pretrained(args.output_dir)

# ------------
# 5. Generate Sample
# ------------
print("\n--- GENERATING SAMPLE ---")
model.to(device)
model.eval()

# Handle prompt construction
prompt_text = args.prompt
if not prompt_text:
    # If no prompt provided, we use the BOS (Beginning Of Sequence) token if available,
    # otherwise we use EOS token (standard GPT-2 practice for "start from scratch")
    start_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    input_ids = torch.tensor([[start_token_id]], device=device)
    print("Prompt: [UNCONDITIONAL] (Starting from BOS/EOS token)")
else:
    input_ids = tokenizer.encode(prompt_text, return_tensors='pt').to(device)
    print(f"Prompt: '{prompt_text}'")

sample_outputs = model.generate(
    input_ids,
    do_sample=True, 
    max_length=300, 
    top_k=50, 
    top_p=0.92, 
    temperature=0.85, # Slightly lower temperature to focus on more probable (coherent) rhyme structures
    num_return_sequences=3, # Generate 3 variants to compare
    pad_token_id=tokenizer.eos_token_id,
    repetition_penalty=1.3, # Stronger penalty to prevent loops
    no_repeat_ngram_size=3  # Prevent repeating same 3-word phrases
)

for i, sample_output in enumerate(sample_outputs):
    decoded = tokenizer.decode(sample_output, skip_special_tokens=True)
    print(f"\nSample {i+1}:")
    print("-" * 20)
    print(decoded)
    print("-" * 50)
