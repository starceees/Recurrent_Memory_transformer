# data/data_loader.py
import torch
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, tokenized_data):
        self.data = tokenized_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "input_ids": self.data[idx]["input_ids"],
            "labels": self.data[idx]["input_ids"]  # Using inputs as labels for language modeling
        }

def collate_fn(batch):
    input_ids = [torch.tensor(item["input_ids"]) for item in batch]
    labels = [torch.tensor(item["labels"]) for item in batch]
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels)
    return {"input_ids": input_ids, "labels": labels}

def get_dataloader(tokenized_data, batch_size=4):
    dataset = TextDataset(tokenized_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return dataloader

if __name__ == "__main__":
    # Test the dataloader with dummy data
    dummy_data = [{"input_ids": [1,2,3,4,5], "labels": [1,2,3,4,5]} for _ in range(10)]
    loader = get_dataloader(dummy_data, batch_size=2)
    for batch in loader:
        print(batch)
        break
