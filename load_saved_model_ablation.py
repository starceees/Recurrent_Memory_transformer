# import torch
# from mem_size_ablation import *
# def load_checkpoint(checkpoint_path: str or Path):
#     """
#     Load a saved checkpoint and reconstruct the model
    
#     Args:
#         checkpoint_path: Path to the .pt checkpoint file
        
#     Returns:
#         model: Loaded model
#         tokenizer: Associated tokenizer
#         config: Reconstructed Config object
#         metrics: Metrics saved with the checkpoint
#     """
#     # Load checkpoint
#     checkpoint = torch.load(checkpoint_path)
    
#     # Reconstruct config
#     config_dict = checkpoint['config']
#     config = Config(
#         input_size=config_dict['input_size'],
#         memory_size=config_dict['memory_size'],
#         batch_size=config_dict['batch_size'],
#         num_epochs=config_dict['num_epochs']
#     )
    
#     # Setup model and load state
#     model, tokenizer = setup_model(config)
#     model.load_state_dict(checkpoint['model_state_dict'])
    
#     # Get metrics
#     metrics = checkpoint['metrics']
    
#     return model, tokenizer, config, metrics

# # Example usage:
# if __name__ == "__main__":
#     # Path to your checkpoint
#     checkpoint_path = "saved_models/mem32_in4096_b1_e3_1733386955/checkpoint_epoch_2.pt"  # Adjust timestamp
    
#     # Load checkpoint
#     model, tokenizer, config, metrics = load_checkpoint(checkpoint_path)
    
#     # Model is ready for inference or continued training
#     model.eval()  # Set to evaluation mode
    
#     # Print loaded config and metrics
#     print("\nLoaded configuration:")
#     for key, value in vars(config).items():
#         print(f"{key}: {value}")
        
#     print("\nCheckpoint metrics:")
#     for key, value in metrics.items():
#         print(f"{key}: {value}")


from mem_size_ablation import (  
    Config,
    setup_model,
    device,
    MemoryCell,
    RecurrentWrapper,
    AutoModelForCausalLM,
    AutoTokenizer,
    torch,
    Path
)

def load_checkpoint(checkpoint_path: str or Path):
    """
    Load a saved checkpoint and reconstruct the model
    
    Args:
        checkpoint_path: Path to the .pt checkpoint file
        
    Returns:
        model: Loaded model
        tokenizer: Associated tokenizer
        config: Reconstructed Config object
        metrics: Metrics saved with the checkpoint
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    
    # Reconstruct config
    config_dict = checkpoint['config']
    config = Config(
        input_size=config_dict['input_size'],
        memory_size=config_dict['memory_size'],
        batch_size=config_dict['batch_size'],
        num_epochs=config_dict['num_epochs']
    )
    
    # Setup model and load state
    model, tokenizer = setup_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Get metrics
    metrics = checkpoint['metrics']
    
    return model, tokenizer, config, metrics

def generate_text(model, tokenizer, prompt, max_new_tokens=50):
    """
    Generate text using the loaded model
    
    Args:
        model: Loaded model
        tokenizer: Associated tokenizer
        prompt: Input text to continue from
        max_new_tokens: Maximum number of new tokens to generate
        
    Returns:
        Generated text
    """
    # Encode prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,  # Changed from max_length
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,  # Enable sampling for more natural text
            temperature=0.7,  # Control randomness (lower = more focused)
            top_p=0.9,  # Nucleus sampling
            repetition_penalty=1.2  # Discourage repetitions
        )
    
    # Decode and return generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

if __name__ == "__main__":
    # Path to your checkpoint
    checkpoint_path = "saved_models/mem32_in4096_b1_e3_1733386955/checkpoint_epoch_2.pt"  # Adjust path as needed
    
    print("Loading model...")
    model, tokenizer, config, metrics = load_checkpoint(checkpoint_path)
    model.eval()  # Set to evaluation mode
    
    print("\nLoaded configuration:")
    for key, value in vars(config).items():
        print(f"{key}: {value}")
        
    print("\nCheckpoint metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    # Interactive text generation
    while True:
        prompt = input("\nEnter your prompt (or 'quit' to exit): ")
        if prompt.lower() == 'quit':
            break
            
        max_tokens = input("Enter maximum number of new tokens to generate (default 50): ")
        max_tokens = int(max_tokens) if max_tokens.isdigit() else 50
        
        print("\nGenerating text...")
        generated_text = generate_text(model, tokenizer, prompt, max_tokens)
        print(f"\nGenerated text:\n{generated_text}")