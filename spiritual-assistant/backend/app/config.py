import os
from dotenv import load_dotenv

# Load environment variables from a .env file if present
load_dotenv()

OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
API_BASE: str | None = os.getenv("OPENAI_BASE_URL")

# A gentle spiritual system prompt
SYSTEM_PROMPT: str = os.getenv(
    "SYSTEM_PROMPT",
    (
        "You are a compassionate Spiritual Guide. Offer calm, practical, non-judgmental guidance. "
        "Draw on universal spiritual wisdom (mindfulness, compassion, gratitude) without promoting any single religion. "
        "Be succinct and supportive. When giving practices, suggest short, achievable steps."
    ),
)

# Retrieval Augmented Generation config (simple local files)
KNOWLEDGE_DIR: str = os.getenv(
    "KNOWLEDGE_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "knowledge"))
)