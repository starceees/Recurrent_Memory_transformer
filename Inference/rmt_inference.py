# Inference/rmt_inference.py
import torch
from model.model import RecurrentMemoryTransformerWrapper, RecurrentMemoryTransformer
from transformers import AutoTokenizer
import argparse

def load_checkpoint(checkpoint_path, model):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    return model

def infer_text(prompt, model, tokenizer, device="cuda" if torch.cuda.is_available() else "cpu", gen_length=100):
    model.eval()
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = inputs["input_ids"].to(device)
    # Generate text using the model's generate method
    generated = model.generate(inputs, length=gen_length, temperature=1.0)
    output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return output_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--prompt", type=str, required=True, help="Input text prompt for inference")
    parser.add_argument("--gen_length", type=int, default=100, help="Length of text to generate")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize the model architecture (adjust hyperparameters as needed)
    model_arch = RecurrentMemoryTransformer(
        dim=512,
        num_tokens=50257,  # using GPT-2 vocabulary size
        depth=12,
        num_memory_tokens=16,
        seq_len=1024,
        causal=True,
        dim_head=64,
        heads=8
    )
    model = RecurrentMemoryTransformerWrapper(model_arch)
    model.to(device)
    
    # Load model checkpoint
    model = load_checkpoint(args.checkpoint, model)
    
    # Load tokenizer (using GPT-2 tokenizer as an example)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    output = infer_text(args.prompt, model, tokenizer, device=device, gen_length=args.gen_length)
    print("Generated Text:")
    print(output)

if __name__ == "__main__":
    main()
