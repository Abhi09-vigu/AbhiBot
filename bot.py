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
import logging
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
import asyncio
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import aiohttp
import aiosqlite
from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)


# -------------------------------------------------------------
# Configuration & Constants
# -------------------------------------------------------------
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_BASE = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3-8b-instruct:free")
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL")  # Optional: sets HTTP-Referer
OPENROUTER_APP_TITLE = os.getenv("OPENROUTER_APP_TITLE")  # Optional: sets X-Title
ADMIN_ID: Optional[int] = None
try:
    ADMIN_ID = int(os.getenv("ADMIN_ID", "").strip()) if os.getenv("ADMIN_ID") else None
except ValueError:
    ADMIN_ID = None

DB_PATH = os.getenv("DATABASE_PATH", "bot.db")

SYSTEM_PROMPT = (
    "You are a helpful, polite Telegram AI assistant. Be concise and clear."
)

logger = logging.getLogger("telegram_ai_bot")
logging.basicConfig(level=logging.INFO)


# -------------------------------------------------------------
# Database Utilities (SQLite via aiosqlite)
# -------------------------------------------------------------
async def init_db() -> None:
    """Initialize the SQLite database and create tables if needed."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                username TEXT,
                first_name TEXT,
                last_seen TEXT NOT NULL,
                message_count INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        await db.commit()


async def upsert_user(user_id: int, username: Optional[str], first_name: Optional[str], *, increment: bool) -> None:
    """Insert or update user record. Does not store message content."""
    now_iso = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        # Try update first
        if increment:
            await db.execute(
                """
                UPDATE users
                SET username = ?, first_name = ?, last_seen = ?, message_count = message_count + 1
                WHERE user_id = ?
                """,
                (username, first_name, now_iso, user_id),
            )
        else:
            await db.execute(
                """
                UPDATE users
                SET username = ?, first_name = ?, last_seen = ?
                WHERE user_id = ?
                """,
                (username, first_name, now_iso, user_id),
            )

        if db.total_changes == 0:
            # Insert if not exists
            await db.execute(
                """
                INSERT INTO users (user_id, username, first_name, last_seen, message_count)
                VALUES (?, ?, ?, ?, ?)
                """,
                (user_id, username, first_name, now_iso, 1 if increment else 0),
            )

        await db.commit()


async def get_stats() -> tuple[int, int, int]:
    """Return total_users, active_users_24h, total_messages."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    cutoff_iso = cutoff.isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT COUNT(*) FROM users") as cur:
            total_users = (await cur.fetchone())[0]
        async with db.execute(
            "SELECT COUNT(*) FROM users WHERE last_seen >= ?",
            (cutoff_iso,),
        ) as cur:
            active_users = (await cur.fetchone())[0]
        async with db.execute("SELECT COALESCE(SUM(message_count), 0) FROM users") as cur:
            total_messages = (await cur.fetchone())[0]
    return int(total_users), int(active_users), int(total_messages)


# -------------------------------------------------------------
# OpenRouter API Client (async via aiohttp)
# -------------------------------------------------------------
class OpenRouterClient:
    """Minimal async client for OpenRouter Chat Completions API."""

    def __init__(self, api_key: str, model: str, base_url: str, site_url: Optional[str], app_title: Optional[str]):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/") + "/chat/completions"
        self.site_url = site_url
        self.app_title = app_title

    async def chat(self, user_message: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        # Optional but recommended headers per OpenRouter docs
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.app_title:
            headers["X-Title"] = self.app_title
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "temperature": 0.7,
        }

        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(self.base_url, headers=headers, json=payload) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"OpenRouter error: {resp.status} - {text[:200]}")
                data = await resp.json()
                # Expected format similar to OpenAI
                try:
                    return data["choices"][0]["message"]["content"].strip()
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError("Invalid response from OpenRouter") from exc

    async def health(self) -> tuple[bool, str]:
        """Check key validity by hitting /models endpoint."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.app_title:
            headers["X-Title"] = self.app_title
        url = self.base_url.rsplit("/", 1)[0] + "/models"
        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers=headers) as resp:
                ok = resp.status == 200
                text = await resp.text()
                return ok, f"{resp.status} - {text[:200]}"


# -------------------------------------------------------------
# Telegram Bot Handlers
# -------------------------------------------------------------
def is_admin(update: Update) -> bool:
    return ADMIN_ID is not None and update.effective_user and update.effective_user.id == ADMIN_ID


async def track_user(update: Update, *, increment: bool) -> None:
    if not update.effective_user:
        return
    user = update.effective_user
    await upsert_user(
        user_id=user.id,
        username=user.username,
        first_name=user.first_name,
        increment=increment,
    )


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Update last_seen but do not increment message_count for /start
    await track_user(update, increment=False)
    text = (
        "Hi! I'm your AI assistant.\n\n"
        "Privacy: I do not store your message content. "
        "I only save your user ID, username, first name, last seen time, and a message counter for usage stats."
    )
    await update.message.reply_text(text)


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Silent reject if not admin
    if not is_admin(update):
        return
    total_users, active_users, total_messages = await get_stats()
    text = (
        f"Total users: {total_users}\n"
        f"Active users (24h): {active_users}\n"
        f"Total messages processed: {total_messages}"
    )
    await update.message.reply_text(text)


async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return

    await track_user(update, increment=True)

    # Typing indicator before generating a reply
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action=ChatAction.TYPING
    )

    # Generate reply via OpenRouter
    try:
        client = OpenRouterClient(
            OPENROUTER_API_KEY,
            OPENROUTER_MODEL,
            OPENROUTER_API_BASE,
            OPENROUTER_SITE_URL,
            OPENROUTER_APP_TITLE,
        )
        reply = await client.chat(update.message.text)
        if not reply:
            reply = "Sorry, something went wrong. Please try again later."
        await update.message.reply_text(reply)
    except Exception as exc:  # noqa: BLE001
        # Log only error summary, never user content
        logger.error("OpenRouter error: %s", str(exc))
        await update.message.reply_text(
            "Sorry, something went wrong. Please try again later."
        )


async def health_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_admin(update):
        return
    client = OpenRouterClient(
        OPENROUTER_API_KEY,
        OPENROUTER_MODEL,
        OPENROUTER_API_BASE,
        OPENROUTER_SITE_URL,
        OPENROUTER_APP_TITLE,
    )
    ok, detail = await client.health()
    status = "OK" if ok else "FAILED"
    await update.message.reply_text(
        f"OpenRouter health: {status}\nDetail: {detail}"
    )


# -------------------------------------------------------------
# Application Entrypoint
# -------------------------------------------------------------
def validate_env() -> None:
    missing = []
    if not TELEGRAM_BOT_TOKEN:
        missing.append("TELEGRAM_BOT_TOKEN")
    if not OPENROUTER_API_KEY:
        missing.append("OPENROUTER_API_KEY")
    if missing:
        vars_str = ", ".join(missing)
        raise RuntimeError(f"Missing required environment variables: {vars_str}")
    # Warn if ADMIN_ID provided but not numeric
    raw_admin = os.getenv("ADMIN_ID")
    if raw_admin and ADMIN_ID is None:
        logging.warning("ADMIN_ID is set but not a numeric Telegram user ID. Admin features will be disabled.")


def build_application() -> Application:
    app = (
        Application.builder()
        .token(str(TELEGRAM_BOT_TOKEN))
        .concurrent_updates(True)
        .build()
    )

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("stats", stats_command))
    app.add_handler(CommandHandler("health", health_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))
    return app


async def async_main() -> None:
    validate_env()
    await init_db()
    application = build_application()
    await application.initialize()
    await application.start()
    await application.updater.start_polling()
    # Run until Ctrl+C or process signal
    await application.updater.idle()
    await application.stop()
    await application.shutdown()


if __name__ == "__main__":
    asyncio.run(async_main())
