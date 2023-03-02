import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

import numpy as np
import os

from config import *
from utils import IntentDataset, bert_tokenize


epochs = 20


tokenizer = BertTokenizer.from_pretrained("cointegrated/rubert-tiny")

train_set = IntentDataset(os.path.join(classification_dataset_path, "train.csv"), tokenizer)
train_loader = DataLoader(train_set, batch_size=3, shuffle=True)

valid_set = IntentDataset(os.path.join(classification_dataset_path, "test.csv"), tokenizer, train_set.classes)
valid_loader = DataLoader(valid_set, batch_size=1)

model = BertForSequenceClassification.from_pretrained(
    "cointegrated/rubert-tiny", num_labels=train_set.num_classes
).cuda().train()
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
loss_function = torch.nn.CrossEntropyLoss()
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * epochs
)


def train_epoch(model, device='cuda'):
    losses = []
    correct_predictions = 0

    for data in train_loader:
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        targets = data["targets"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        preds = torch.argmax(outputs.logits, dim=1)
        loss = loss_function(outputs.logits, targets)

        correct_predictions += torch.sum(preds == targets)

        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    mean_accuracy = correct_predictions.double() / len(train_set)
    mena_loss = np.mean(losses)
    return mean_accuracy, mena_loss


def eval_epoch(model, device='cuda'):
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for data in valid_loader:
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            targets = data["targets"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            preds = torch.argmax(outputs.logits, dim=1)
            loss = loss_function(outputs.logits, targets)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

    val_acc = correct_predictions.double() / len(valid_set)
    val_loss = np.mean(losses)
    return val_acc, val_loss


for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}')
    model = model.train()
    train_acc, train_loss = train_epoch(model)
    print(f'Train loss {train_loss} accuracy {train_acc}')

    model = model.eval()
    val_acc, val_loss = eval_epoch(model)
    print(f'Val loss {val_loss} accuracy {val_acc}')
    print('-' * 10)

model = model.eval()
model.id2class = train_set.id2class
model.tokenizer = tokenizer


def classification_inference(text: str, device="cuda") -> str:
    encoding = bert_tokenize(model.tokenizer, text)

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).logits.detach().cpu()

    return model.id2class[torch.argmax(out, dim=1).item()]


for i in valid_set:
    print(i['text'], classification_inference(i['text']))

torch.save(model, weight_path)
