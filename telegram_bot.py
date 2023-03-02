from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

from config import *
from engine import prepared_messages, classification_inference, base_answer, gpt_inference


user_intents = {}
message_history = {}


def log_message(name, text):
    global message_history
    message_history[name].append(text)
    message_history[name] = message_history[name][-HISTORY_SIZE:]
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

    if name not in user_intents.keys():
        user_intents[name] = []
        message_history[name] = []

    log_message(name, text)
    intent = classification_inference(text)

    services_hist = [i for i in user_intents[name] if i in services_list]
    if intent in services_list and intent not in services_hist:
        ans = base_answer() + prepared_messages[intent]
    elif intent == "trade" and len(services_hist):
        intent = services_hist[-1] + "_trade"
        ans = prepared_messages[intent]
    elif intent in ["name", "description"]:
        ans = prepared_messages[intent]
    else:
        intent = "other"
        input_text = "- " + "\n- ".join(message_history[name]) + "\n-"
        ans = gpt_inference(input_text)

    user_intents[name].append(intent)
    user_intents[name] = user_intents[name][-HISTORY_SIZE:]

    log_message(name, ans)

    await context.bot.send_message(
        update.effective_chat.id, text=ans
    )


if __name__ == '__main__':
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler('start', start))
    app.add_handler(MessageHandler(filters.TEXT, conversation))
    app.run_polling()
