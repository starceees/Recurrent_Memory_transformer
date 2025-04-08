# data/data_ingest.py
import os
import json
from transformers import AutoTokenizer

def load_and_preprocess_data(data_dir):
    """
    Load and preprocess data from the specified directory.
    This example assumes a JSONL file format with one JSON object per line.
    Optionally, additional embeddings (e.g. RoPE) can be applied here.
    """
    data_path = os.path.join(data_dir, "data.jsonl")
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            data.append(entry["text"])
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenized_data = [tokenizer(text, truncation=True, max_length=1024, padding="max_length") for text in data]
    return tokenized_data

if __name__ == "__main__":
    data = load_and_preprocess_data("data")
    print(f"Loaded {len(data)} entries from data.")
