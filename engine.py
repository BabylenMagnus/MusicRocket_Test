import random
import yaml
import torch

from config import *
from  utils import bert_tokenize

from transformers import AutoTokenizer, AutoModelForCausalLM, BertTokenizer, BatchEncoding


with open(prepared_messages_path, "r") as t:
    prepared_messages = yaml.safe_load(t)

base_answer = lambda: random.choice(prepared_messages["base_answers"]) + ": "

device = "cuda" if torch.cuda.is_available() else "cpu"
classification_model = torch.load(weight_path).to(device).eval()

# load gpt model
gpt_model = AutoModelForCausalLM.from_pretrained("inkoziev/rugpt_chitchat").to(device).eval()
gpt_model.tokenizer = AutoTokenizer.from_pretrained("inkoziev/rugpt_chitchat")
gpt_model.tokenizer.add_special_tokens({'bos_token': '<s>', 'eos_token': '</s>', 'pad_token': '<pad>'})


def classification_inference(text: str) -> str:
    """
    Классифицирует текст одним из классов, которые есть в train.csv
    """
    encoding = bert_tokenize(classification_model.tokenizer, text)

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    with torch.no_grad():
        out = classification_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).logits.detach().cpu()

    return classification_model.id2class[torch.argmax(out, dim=1).item()]


def gpt_inference(input_text):
    """
    GPT модель, которая может разнообразить диалог
    """
    with torch.no_grad():
        encoded_prompt = gpt_model.tokenizer.encode(
            input_text, add_special_tokens=False, return_tensors="pt"
        ).to(device)
        output_sequences = gpt_model.generate(
            input_ids=encoded_prompt, max_length=200, num_return_sequences=1,
            pad_token_id=gpt_model.tokenizer.pad_token_id
        )
        text = gpt_model.tokenizer.decode(
            output_sequences[0].tolist(), clean_up_tokenization_spaces=True
        )[len(input_text) + 1:]
    text = text[: text.find('</s>')]
    # sometimes generates more than 1 line, there need to be a limit
    text = text[: text.find('\n')] if text.find('\n') > 0 else text
    return text
