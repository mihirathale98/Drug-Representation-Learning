from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pickle


class MLMDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.sep = self.tokenizer.sep_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        item = list(item)
        item = self.sep.join(item)
        item = self.tokenizer(item, return_tensors="pt", padding="max_length", truncation=True)
        item = {k: v.squeeze() for k, v in item.items()}
        return item


