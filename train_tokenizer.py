import os
import argparse
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

def train_tokenizer(input_file, output_file, vocab_size):
    # Initialize a tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    # Initialize a trainer
    trainer = BpeTrainer(
        vocab_size=vocab_size, 
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "<|endoftext|>"]
    )

    # Train the tokenizer
    files = [input_file]
    tokenizer.train(files, trainer)

    # Save the tokenizer
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    tokenizer.save(output_file)

    print(f"Tokenizer trained on {input_file} and saved to {output_file}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a custom BPE tokenizer')
    parser.add_argument('--input', type=str, default='assets/input/input2.txt', help='Path to input text file')
    parser.add_argument('--output', type=str, default='assets/tiktoken_model/tokenizer.json', help='Path to save tokenizer.json')
    parser.add_argument('--vocab_size', type=int, default=5000, help='Vocabulary size')
    args = parser.parse_args()

    train_tokenizer(args.input, args.output, args.vocab_size)
