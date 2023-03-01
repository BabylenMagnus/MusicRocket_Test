from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

from config import *
from engine import prepared_messages, classification_inference, base_answer, gpt_inference


user_intents = {}
message_history = {}


def log_message(name, text):
    global message_history
    message_history[name].append(text)
    message_history[name] = message_history[name][-6:]
    with open(f"logging/{name}.txt", "a") as t:
        t.write("\n" + text)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    name = update.effective_chat.username
    user_intents[name] = ["other"]
    message_history[name] = [prepared_messages["start"]]

    await context.bot.send_message(
        update.effective_chat.id, text=prepared_messages["start"]
    )


async def conversation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    name = update.effective_chat.username
    text = update.message.text

    log_message(name, text)

    intent = classification_inference(text)
    k = [i for i in user_intents[name] if i != "other" and "trade" not in i]
    if intent == "trade" and len(k) == 0:
        intent = "other"
    elif intent == "trade":
        intent = k[-1] + "_trade"
    elif intent in user_intents[name]:
        intent = "other"

    intent = "other" if intent == user_intents[name][-1] else intent

    if "trade" in intent:
        ans = prepared_messages[intent]
    elif intent == "other":
        input_text = "- " + "\n- ".join(message_history[name]) + "\n-"
        ans = gpt_inference(input_text)
    else:
        ans = base_answer() + prepared_messages[intent]

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
