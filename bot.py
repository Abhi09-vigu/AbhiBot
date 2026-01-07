"""
Telegram AI Chatbot using python-telegram-bot (async) + OpenRouter API.

Features:
- /start command greeting
- Forwards user messages to OpenRouter and replies with AI output
- Uses environment variables for secrets (TELEGRAM_BOT_TOKEN, OPENROUTER_API_KEY)
- Graceful error handling with a friendly fallback message
- Async/await throughout

To run:
1) Create and fill in .env with your tokens
2) Install dependencies from requirements.txt
3) python bot.py
"""

import asyncio
import logging
import os
import re
from typing import Optional

import aiohttp
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# -------------------------
# Configuration and Logging
# -------------------------

# Load environment variables from .env if present
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Choose a free OpenRouter model
# Options include "mistralai/mistral-7b-instruct" or "meta-llama/llama-3-8b-instruct"
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct")

# Configure logging for production-friendly visibility
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("telegram_ai_bot")


# -------------------------
# OpenRouter API Client
# -------------------------

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


async def call_openrouter(prompt: str, session: aiohttp.ClientSession) -> Optional[str]:
    """Call the OpenRouter chat completions API with a simple user message.

    Args:
        prompt: The user's message to send to the AI model.
        session: An existing aiohttp ClientSession for connection reuse.

    Returns:
        The assistant's reply text, or None if something goes wrong.
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        # Optional headers to identify your app/website for rate-limit courtesy
        "HTTP-Referer": "https://example.com",
        "X-Title": "Abhi",
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful AI assistant named Abhi. Keep responses concise and friendly. "
                    "Avoid numbered lists or bullet points unless the user explicitly asks. "
                    "Use real newlines (not literal \\n) and avoid extraneous whitespace."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        # Reasonable defaults
        "temperature": 0.7,
        "max_tokens": 512,
    }

    try:
        async with session.post(OPENROUTER_API_URL, json=payload, headers=headers, timeout=60) as resp:
            if resp.status != 200:
                text = await resp.text()
                logger.error("OpenRouter HTTP %s: %s", resp.status, text)
                return None
            data = await resp.json()
    except asyncio.TimeoutError:
        logger.exception("Timeout when calling OpenRouter")
        return None
    except aiohttp.ClientError:
        logger.exception("Network error when calling OpenRouter")
        return None
    except Exception:  # noqa: BLE001 - broad catch for production resilience
        logger.exception("Unexpected error when calling OpenRouter")
        return None

    # Extract assistant message safely
    try:
        choices = data.get("choices", [])
        if not choices:
            logger.error("OpenRouter response missing choices: %s", data)
            return None
        message = choices[0].get("message", {})
        content = message.get("content")
        if not content:
            logger.error("OpenRouter message missing content: %s", message)
            return None
        return content
    except Exception:
        logger.exception("Failed to parse OpenRouter response")
        return None


# -------------------------
# Formatting helpers
# -------------------------

def format_response(text: str) -> str:
    """Normalize model output:
    - Convert literal "\\n" to real newlines
    - Collapse excessive blank lines
    - Trim surrounding whitespace
    """
    if not text:
        return ""
    t = text.replace("\r\n", "\n")
    t = t.replace("\\n", "\n")
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


async def send_chunked(context: ContextTypes.DEFAULT_TYPE, chat_id: int, text: str) -> None:
    """Send text to Telegram, splitting into safe chunks if needed."""
    max_len = 4096
    if len(text) <= max_len:
        await context.bot.send_message(chat_id=chat_id, text=text)
        return
    start = 0
    while start < len(text):
        end = min(start + max_len, len(text))
        split = text.rfind("\n", start, end)
        if split == -1 or split <= start + 100:
            split = end
        await context.bot.send_message(chat_id=chat_id, text=text[start:split])
        start = split


# -------------------------
# Telegram Handlers
# -------------------------

async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /start command with a friendly greeting."""
    if update.effective_chat is None:
        return
    await send_chunked(
        context,
        update.effective_chat.id,
        "Hello! I am Abhi, your AI assistant. Ask me anything.",
    )


async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle any text message: send to OpenRouter and reply with AI output."""
    if update.message is None or update.effective_chat is None:
        return

    user_text = update.message.text or ""

    # Create a short-lived HTTP session per request to simplify lifecycle
    async with aiohttp.ClientSession() as session:
        reply_text = await call_openrouter(user_text, session)

    # Format and ensure non-empty text is sent back to Telegram
    reply_text = format_response(reply_text or "")
    if not reply_text:
        reply_text = "Sorry, something went wrong. Please try again."
    await send_chunked(context, update.effective_chat.id, reply_text)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log errors from the dispatcher without crashing the bot."""
    logger.exception("Exception while handling an update: %s", context.error)
    try:
        if isinstance(update, Update) and update.effective_chat is not None:
            await send_chunked(context, update.effective_chat.id, "Sorry, something went wrong. Please try again.")
    except Exception:
        # Avoid raising inside error handler
        pass


# -------------------------
# Application Entrypoint
# -------------------------

def main() -> None:
    """Build and run the Telegram bot application."""
    # Basic validation for required secrets
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN environment variable.")
    if not OPENROUTER_API_KEY:
        raise RuntimeError("Missing OPENROUTER_API_KEY environment variable.")

    # Build application
    application: Application = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .build()
    )

    # Register handlers
    application.add_handler(CommandHandler("start", start_handler))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))

    # Global error handler
    application.add_error_handler(error_handler)

    # Run the bot until Ctrl+C or process kill (synchronous API in PTB v21)
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
