import numpy as np
import os
import sys
import tqdm
import torch
import datasets
import math
import wandb
import time
from pathlib import Path
from matplotlib import pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from itertools import chain
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from modeling_rmt.language_modeling import MemoryCell, RecurrentWrapper
from torch.optim import AdamW
from typing import Dict, Any


device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Configuration class to handle all parameters
class Config:
    def __init__(self, input_size: int, memory_size: int, batch_size: int, num_epochs: int):
        self.input_size = input_size
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.model_name = 'gpt2'
        self.block_size = 1024 - 2 * memory_size  # Adjusted as per original code
        self.n_segments = math.ceil(input_size / self.block_size)
        self.history_size = (self.n_segments - 1) * self.block_size
        self.learning_rate = 1e-4
        self.train_steps = 100
        self.eval_steps = 100
        
        # Create a unique run name for saving models
        self.run_name = f"mem{memory_size}_in{input_size}_b{batch_size}_e{num_epochs}_{int(time.time())}"
        self.save_dir = Path("saved_models") / self.run_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

def group_texts(examples, block_size, history_size=None):
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])

    if history_size is None:
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
    else:
        result = {
            k: [t[max({0, i - history_size}) : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
    result["labels"] = result["input_ids"].copy()
    return result

def collate_fn(batch, id_pad_value, block_size):
    input_ids = [torch.tensor(b['input_ids'][::-1]) for b in batch]
    labels = [torch.tensor(b['labels'][::-1]) for b in batch]
    attention_mask = [torch.tensor(b['attention_mask'][::-1]) for b in batch]
    input_ids = pad_sequence(input_ids, padding_value=id_pad_value).T.flip(1)
    labels = pad_sequence(labels, padding_value=-100).T.flip(1)
    attention_mask = pad_sequence(attention_mask, padding_value=0).T.flip(1)

    collated = {'input_ids': input_ids,
                'labels': labels,
                'attention_mask': attention_mask}

    if input_ids.shape[1] != block_size:
        labels_mask = torch.ones_like(input_ids, dtype=bool)
        labels_mask[:, :-block_size] = False
        collated['labels_mask'] = labels_mask

    return collated

def setup_model(config: Config):
    model = AutoModelForCausalLM.from_pretrained(config.model_name)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    print(f"Running with config: {vars(config)}")
    
    cell = MemoryCell(model, num_mem_tokens=config.memory_size)
    model = RecurrentWrapper(cell,
                           segment_size=config.block_size,
                           max_n_segments=config.n_segments)
    model.to(device)
    return model, tokenizer

def compute_perplexity(loss):
    """Compute perplexity from loss"""
    return torch.exp(loss).item()

def prepare_dataset(config: Config, tokenizer):
    def tokenize_function(examples):
        return tokenizer(examples["text"])

    id_pad_value = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    task_name = 'wikitext-2-v1'
    raw_datasets = datasets.load_dataset('wikitext', task_name)
    column_names = raw_datasets["train"].column_names
    
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        desc="Running tokenizer on dataset",
    )
    
    # Custom collate function with closure over pad_value and block_size
    collate_with_config = lambda batch: collate_fn(batch, id_pad_value, config.block_size)
    
    train_dataset = tokenized_datasets["train"].map(
        lambda x: group_texts(x, config.block_size, config.history_size),
        batched=True,
        desc=f"Grouping train in chunks of {config.block_size}"
    )
    
    valid_dataset = tokenized_datasets["validation"].map(
        lambda x: group_texts(x, config.block_size, config.history_size),
        batched=True,
        desc=f"Grouping validation in chunks of {config.block_size}"
    )
    
    test_dataset = tokenized_datasets["test"].map(
        lambda x: group_texts(x, config.block_size, config.history_size),
        batched=True,
        desc=f"Grouping test in chunks of {config.block_size}"
    )
    
    train_rnd_generator = torch.Generator()
    train_rnd_generator.manual_seed(42)
    
    dataloaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            collate_fn=collate_with_config,
            shuffle=True,
            drop_last=False,
            generator=train_rnd_generator,
            pin_memory=True
        ),
        'valid': DataLoader(
            valid_dataset,
            batch_size=config.batch_size,
            collate_fn=collate_with_config,
            shuffle=False,
            drop_last=True,
            pin_memory=True
        ),
        'test': DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            collate_fn=collate_with_config,
            shuffle=False,
            drop_last=True,
            pin_memory=True
        )
    }
    
    return dataloaders

def train_epoch(model, dataloader, optimizer, config: Config, epoch: int):
    model.train()
    total_loss = 0
    total_perplexity = 0
    start_time = time.time()
    
    train_gen = iter(dataloader)
    
    for step in range(len(dataloader)):
    # for step in tqdm.tqdm(range(1), desc=f"Evaluating on train"):

        step_start = time.time()
        optimizer.zero_grad()
        
        batch = next(train_gen)
        batch = {k: v.to(device) for k, v in batch.items()}
        
        out = model(**batch)
        loss = out.loss
        
        loss.backward()
        optimizer.step()
        
        perplexity = compute_perplexity(loss)
        total_loss += loss.item()
        total_perplexity += perplexity
        
        batch_time = time.time() - step_start
        
        # Log metrics
        wandb.log({
            'train/loss': loss.item(),
            'train/perplexity': perplexity,
            'train/batch_time': batch_time,
            'epoch': epoch,
            'step': step
        })
    
    epoch_time = time.time() - start_time
    metrics = {
        'train/epoch_loss': total_loss / config.train_steps,
        'train/epoch_perplexity': total_perplexity / config.train_steps,
        'train/epoch_time': epoch_time
    }
    wandb.log(metrics)
    return metrics

def validate(model, dataloader, config: Config, split='valid'):
    model.eval()
    total_loss = 0
    total_perplexity = 0
    start_time = time.time()
    
    valid_gen = iter(dataloader)
    
    for step in range(len(dataloader)):
    # for step in tqdm.tqdm(range(1), desc=f"Evaluating on {split}"):

        batch = next(valid_gen)
        batch = {k: v.to(device) for k, v in batch.items()}
        
        with torch.no_grad():
            out = model(**batch)
        loss = out.loss
        
        perplexity = compute_perplexity(loss)
        total_loss += loss.item()
        total_perplexity += perplexity
        
        # Log step metrics
        wandb.log({
            f'{split}/step_loss': loss.item(),
            f'{split}/step_perplexity': perplexity,
        })
    
    metrics = {
        f'{split}/loss': total_loss / config.eval_steps,
        f'{split}/perplexity': total_perplexity / config.eval_steps,
        f'{split}/time': time.time() - start_time
    }
    wandb.log(metrics)
    return metrics

def save_checkpoint(model, optimizer, epoch, metrics, config: Config):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'config': vars(config),
        'metrics': metrics
    }
    save_path = config.save_dir / f'checkpoint_epoch_{epoch}.pt'
    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint to {save_path}")

# def main():
#     # Initialize wandb
#     wandb.init(
#         project="memory-transformer",
#         config={
#             "memory_sizes": [2, 4, 8, 16, 32, 64, 128]  # For sweeping
#         }
#     )
    
#     # Get configuration from wandb if sweeping, otherwise use defaults
#     memory_size = wandb.config.get("memory_size", 128)
#     config = Config(
#         input_size=4096,
#         memory_size=memory_size,
#         batch_size=1,
#         num_epochs=3
#     )
    
#     # Setup model and datasets
#     model, tokenizer = setup_model(config)
#     dataloaders = prepare_dataset(config, tokenizer)
#     optimizer = AdamW(params=model.parameters(), lr=config.learning_rate)
    
#     # Training loop
#     for epoch in range(config.num_epochs):
#         print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        
#         # Train
#         train_metrics = train_epoch(model, dataloaders['train'], optimizer, config, epoch)
        
#         # Validate
#         valid_metrics = validate(model, dataloaders['valid'], config, split='valid')
        
#         # Save checkpoint
#         all_metrics = {**train_metrics, **valid_metrics}
#         save_checkpoint(model, optimizer, epoch, all_metrics, config)
    
#     # Final test evaluation
#     test_metrics = validate(model, dataloaders['test'], config, split='test')
#     print(f"\nFinal test metrics: {test_metrics}")
    
#     wandb.finish()

# if __name__ == "__main__":
#     main()


def main():
    # Define sweep configuration
    sweep_config = {
        'method': 'grid',  # Using grid search since we have specific memory sizes
        'name': 'memory_size_sweep',
        'metric': {
            'name': 'test/perplexity',  
            'goal': 'minimize'         
        },
        'parameters': {
            'memory_size': {
                'values': [2, 4, 8, 16, 32, 64, 128]
            }
        }
    }

    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project="memory-transformer")

    def sweep_train():
        # Initialize wandb for this run
        wandb.init()
        
        # Get memory size from sweep
        memory_size = wandb.config.memory_size
        
        # Create config with current memory size
        config = Config(
            input_size=4096,
            memory_size=memory_size,
            batch_size=1,
            num_epochs=3
        )
        wandb.config.update({"run_name": config.run_name})
        wandb.run.summary["run_name"] = config.run_name
        
        # Setup model and datasets
        model, tokenizer = setup_model(config)
        dataloaders = prepare_dataset(config, tokenizer)
        optimizer = AdamW(params=model.parameters(), lr=config.learning_rate)
        
        # Training loop
        for epoch in range(config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
            
            # Train
            train_metrics = train_epoch(model, dataloaders['train'], optimizer, config, epoch)
            
            # Validate
            valid_metrics = validate(model, dataloaders['valid'], config, split='valid')
            
            # Save checkpoint
            all_metrics = {**train_metrics, **valid_metrics}
            save_checkpoint(model, optimizer, epoch, all_metrics, config)
            
            # Run test evaluation after each epoch to track progress
            test_metrics = validate(model, dataloaders['test'], config, split='test')
            print(f"Epoch {epoch + 1} test perplexity: {test_metrics['test/perplexity']:.2f}")
        
        # Final test evaluation
        final_test_metrics = validate(model, dataloaders['test'], config, split='test')
        print(f"\nFinal test metrics: {final_test_metrics}")
        
        # Log final test perplexity as the key metric for comparison
        wandb.run.summary['final_test_perplexity'] = final_test_metrics['test/perplexity']
        
        wandb.finish()

    # Run the sweep
    wandb.agent(sweep_id, function=sweep_train, count=7)  # 7 runs for 7 different memory sizes

if __name__ == "__main__":
    main()