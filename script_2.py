import wandb
import argparse
import os
import sys
import torch
import json
import datasets
from datetime import datetime
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from itertools import chain
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modeling_rmt.language_modeling import MemoryCell, RecurrentWrapper

class CustomDataCollator:
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.block_size = args.input_size - 2 * args.memory_size
    
    def __call__(self, batch):
        id_pad_value = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        input_ids = [torch.tensor(b['input_ids'][::-1]) for b in batch]
        labels = [torch.tensor(b['labels'][::-1]) for b in batch]
        attention_mask = [torch.tensor(b['attention_mask'][::-1]) for b in batch]
        
        input_ids = pad_sequence(input_ids, padding_value=id_pad_value).T.flip(1)
        labels = pad_sequence(labels, padding_value=-100).T.flip(1)
        attention_mask = pad_sequence(attention_mask, padding_value=0).T.flip(1)

        collated = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }

        if input_ids.shape[1] != self.block_size:
            labels_mask = torch.ones_like(input_ids, dtype=bool)
            labels_mask[:, :-self.block_size] = False
            collated['labels_mask'] = labels_mask

        return collated

def setup_model(args):
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["c_attn"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    block_size = args.input_size - 2 * args.memory_size
    cell = MemoryCell(model, num_mem_tokens=args.memory_size)
    rmt_model = RecurrentWrapper(cell,
                               segment_size=block_size,
                               max_n_segments=args.n_segments)
    
    return rmt_model, tokenizer

def prepare_dataset(tokenizer, args):
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
                k: [t[max(0, i - history_size) : i + block_size] 
                   for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
        result["labels"] = result["input_ids"].copy()
        return result
    
    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])
    
    raw_datasets = datasets.load_dataset('wikitext', 'wikitext-2-v1')
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]
    
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        desc="Running tokenizer on dataset",
    )
    
    block_size = args.input_size - 2 * args.memory_size
    history_size = (args.n_segments - 1) * block_size
    
    train_dataset = tokenized_datasets["train"].map(
        lambda x: group_texts(x, block_size, history_size),
        batched=True,
        desc=f"Grouping train in chunks of {block_size}"
    )
    
    valid_dataset = tokenized_datasets["validation"].map(
        lambda x: group_texts(x, block_size, history_size),
        batched=True,
        desc=f"Grouping valid in chunks of {block_size}"
    )
    
    return train_dataset, valid_dataset

def setup_sweep_batch_size():
    return {
        'method': 'grid',
        'metric': {
            'name': 'eval_loss',
            'goal': 'minimize'
        },
        'parameters': {
            'batch_size': {'values': [1, 2, 4, 8]},
            'input_size': {'value': 1024}
        }
    }

def setup_sweep_n_segments(best_batch_size):
    return {
        'method': 'grid',
        'metric': {
            'name': 'eval_loss',
            'goal': 'minimize'
        },
        'parameters': {
            'n_segments': {'values': [2, 4, 6, 8]},
            'input_size': {'value': 1024},
            'batch_size': {'value': best_batch_size}
        }
    }

def train_model(config=None):
    with wandb.init(config=config) as run:
        config = wandb.config
        
        default_params = {
            'model_name': 'gpt2',
            'memory_size': 4,
            'lora_r': 16,
            'lora_alpha': 32,
            'lora_dropout': 0.1,
            'learning_rate': 1e-4,
            'batch_size': config.get('batch_size', 1),
            'n_segments': config.get('n_segments', 2),
            'input_size': config.get('input_size', 1024)
        }
        
        args = argparse.Namespace(**default_params)
        args.output_dir = f'./results/run_{run.id}'
        args.eval_during_training = True
        args.eval_steps = 100
        args.mode = 'train'
        
        try:
            model, tokenizer = setup_model(args)
            train_dataset, valid_dataset = prepare_dataset(tokenizer, args)
            
            total_samples = len(train_dataset)
            steps_per_epoch = total_samples // args.batch_size
            
            training_args = TrainingArguments(
                output_dir=args.output_dir,
                save_strategy="epoch",
                save_total_limit=2,
                evaluation_strategy="epoch",
                logging_dir=f"{args.output_dir}/logs",
                learning_rate=args.learning_rate,
                per_device_train_batch_size=args.batch_size,
                per_device_eval_batch_size=args.batch_size,
                num_train_epochs=10,
                weight_decay=0.01,
                warmup_steps=steps_per_epoch,
                fp16=True,
                report_to="wandb",
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                save_safetensors=False
            )
            
            model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=valid_dataset,
                data_collator=CustomDataCollator(args, tokenizer)
            )
            
            train_result = trainer.train()
            
            metrics = {
                "final_train_loss": train_result.training_loss,
                "total_steps": train_result.global_step
            }
            
            if hasattr(trainer.state, 'best_metric'):
                metrics["best_eval_loss"] = trainer.state.best_metric
            
            wandb.log(metrics)
            
            output_id = f"rmt_{args.model_name}_{args.input_size}_{args.memory_size}_{args.n_segments}"
            model_path = os.path.join(args.output_dir, f"{output_id}.bin")
            torch.save(model.state_dict(), model_path)
            
            artifact = wandb.Artifact(
                name=f"model_{run.id}",
                type="model",
                description=f"RMT model with input_size={args.input_size}, batch_size={args.batch_size}, n_segments={args.n_segments}"
            )
            artifact.add_file(model_path)
            run.log_artifact(artifact)
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            wandb.log({"error": str(e)})
            raise e

def get_best_params(project_name, sweep_id):
    api = wandb.Api()
    sweep = api.sweep(f"{project_name}/{sweep_id}")
    best_run = min(sweep.runs, key=lambda run: run.summary.get('best_eval_loss', float('inf')))
    return best_run.config

def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f"sequential_sweeps_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    project_name = "rmt-sequential-sweeps"
    entity_name = wandb.Api().default_entity
    
    # print("Starting batch_size sweep...")
    # sweep_id = wandb.sweep(
    #     setup_sweep_batch_size(),
    #     project=project_name,
    #     entity=entity_name
    # )
    # wandb.agent(sweep_id, train_model, count=4)
    # best_batch_size = get_best_params(f"{entity_name}/{project_name}", sweep_id)['batch_size']
    
    # print(f"Best batch_size: {best_batch_size}")
    
    print("Starting n_segments sweep...")
    sweep_id = wandb.sweep(
        setup_sweep_n_segments(1),
        project=project_name,
        entity=entity_name
    )
    wandb.agent(sweep_id, train_model, count=4)
    final_params = get_best_params(f"{entity_name}/{project_name}", sweep_id)
    
    final_config = {
        'input_size': 1024,
        'best_batch_size': best_batch_size,
        'best_n_segments': final_params['n_segments']
    }
    
    print("Final best parameters:", final_config)
    with open(os.path.join(results_dir, 'best_params.json'), 'w') as f:
        json.dump(final_config, f, indent=2)

if __name__ == "__main__":
    main()