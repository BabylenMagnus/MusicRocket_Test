from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

import yaml
import torch
import random
import pytils

from config import *

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# базовые манипуляции
answer = lambda: random.choice([
    "Специально для вас: ", "Не пропустите последнюю неделю скидок: ",
    "Цена ниже чем у конкурентов: ", "Искусство требует жертв, к счастью это не про наши цены: "
])

with open("messages.yaml", "r") as t:
    messages = yaml.safe_load(t)

model = torch.load(weight_path).cuda().eval()

user_intents = {}

tokenizer = AutoTokenizer.from_pretrained("inkoziev/rugpt_chitchat")
tokenizer.add_special_tokens({'bos_token': '<s>', 'eos_token': '</s>', 'pad_token': '<pad>'})
gpt_model = AutoModelForCausalLM.from_pretrained("inkoziev/rugpt_chitchat").cuda().eval()

message_history = {}


def log_message(name, text):
    global message_history
    message_history[name].append(text)
    message_history[name] = message_history[name][-6:]
    with open(f"logging/{name}.txt", "a") as t:
        t.write("\n" + text)


def classification(text, device='cuda'):
    encoding = model.tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask).logits.detach().cpu()

    return model.id2class[torch.argmax(out, dim=1).item()]


def inference_gpt(input_text):
    encoded_prompt = tokenizer.encode(input_text, add_special_tokens=False, return_tensors="pt").to("cuda")

    output_sequences = gpt_model.generate(
        input_ids=encoded_prompt, max_length=200, num_return_sequences=1, pad_token_id=tokenizer.pad_token_id
    )

    text = tokenizer.decode(output_sequences[0].tolist(), clean_up_tokenization_spaces=True)[len(input_text) + 1:]
    text = text[: text.find('</s>')]
    text = text[: text.find('\n')] if text.find('\n') > 0 else text
    return text


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    name = update.effective_chat.username
    user_intents[name] = ["other"]
    message_history[name] = [messages["start"]]

    await context.bot.send_message(
        update.effective_chat.id, text=messages["start"]
    )


async def conversation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    name = update.effective_chat.username
    text = update.message.text

    log_message(name, text)

    intent = classification(text)
    if intent == "trade" and len(user_intents[name]) == 1:
        intent = "other"
    elif intent == "trade":
        intent = [i for i in user_intents[name] if i != "other" and "trade" not in i][-1] + "_trade"
    elif intent in user_intents[name]:
        intent = "other"

    intent = "other" if intent == user_intents[name][-1] else intent

    if "trade" in intent:
        ans = messages[intent]
    elif intent == "other":
        input_text = "- " + "\n- ".join(message_history[name]) + "\n-"
        ans = inference_gpt(input_text)
    else:
        ans = answer() + messages[intent]

    user_intents[name].append(intent)
    user_intents[name] = user_intents[name][-5:]

    log_message(name, ans)

    await context.bot.send_message(
        update.effective_chat.id, text=ans
    )


if __name__ == '__main__':
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler('start', start))
    app.add_handler(MessageHandler(filters.TEXT, conversation))
    app.run_polling()
