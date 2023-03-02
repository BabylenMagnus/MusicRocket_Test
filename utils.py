import torch
from transformers import BertTokenizer, BatchEncoding
from torch.utils.data import Dataset

import pandas as pd


def bert_tokenize(tokenizer: BertTokenizer, text: str) -> BatchEncoding:
    return tokenizer.encode_plus(
        text, add_special_tokens=True, max_length=512, return_token_type_ids=False,
        padding='max_length', return_attention_mask=True, return_tensors='pt'
    )


class IntentDataset(Dataset):
    def __init__(self, csv_path: str, tokenizer: BertTokenizer, classes: list[str] = None):
        df = pd.read_csv(csv_path, sep=";")
        self.classes = sorted(df.intent.unique()) if classes is None else classes

        self.id2class = {i: x for i, x in enumerate(self.classes)}
        self.class2id = {x: i for i, x in enumerate(self.classes)}

        self.labels = [self.class2id[x] for x in df.intent]
        self.objects = list(df.text)
        self.num_classes = len(self.classes)

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.objects)

    def __getitem__(self, i):
        text = str(self.objects[i])
        encoding = bert_tokenize(self.tokenizer, text)

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(self.labels[i], dtype=torch.long)
        }
