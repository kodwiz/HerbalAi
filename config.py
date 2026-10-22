"""Central configuration. Secrets come from environment variables or Streamlit secrets only."""
from __future__ import annotations

import os
from pathlib import Path

try:  # python-dotenv is optional at runtime; .env is only a local convenience
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
HERBS_PATH = DATA_DIR / "herbs.json"
PRODUCTS_PATH = DATA_DIR / "products.json"
SAFETY_RULES_PATH = DATA_DIR / "safety_rules.json"

# Professional-herbalist knowledge base (herb monographs, minor-condition protocols, graduated safety rules).
HERBALIST_HERBS_PATH = DATA_DIR / "herbalist_herbs.json"
HERBALIST_CONDITIONS_PATH = DATA_DIR / "herbalist_conditions.json"
HERBALIST_RULES_PATH = DATA_DIR / "herbalist_rules.json"
HERBALIST_EXAMPLES_PATH = DATA_DIR / "herbalist_examples.json"

APP_NAME = "Herbal Wellness AI"

# Topics shown as quick-start buttons in the sidebar (herbalist mode).
HERBALIST_TOPICS = {
    "digestion": "Digestion",
    "respiratory": "Cough, cold & throat",
    "sleep_stress": "Sleep & stress",
    "pain": "Aches & headaches",
    "womens_health": "Women's health",
    "skin": "Skin",
    "hair": "Hair & scalp",
    "immunity": "Immunity & fever",
}

# Concern categories shown in the sidebar and used by the assessment engine.
CATEGORIES = {
    "skin": "Skin",
    "hair": "Hair & scalp",
    "digestion": "Digestion",
    "sleep_stress": "Sleep & stress",
    "nutrition": "Nutrition",
    "immunity": "Immunity",
    "general": "General wellness",
}

# Herbs/products must be adults unless the data explicitly states a lower minimum age.
DEFAULT_MIN_AGE_YEARS = 18
CONVERSATION_TURNS_SENT_TO_MODEL = 12


def get_secret(name: str, default: str | None = None) -> str | None:
    """Read a setting from the environment, then from Streamlit secrets if available."""
    value = os.environ.get(name)
    if value:
        return value
    try:  # Streamlit secrets are optional and only exist inside a running app
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return default


def api_key() -> str | None:
    key = get_secret("ANTHROPIC_API_KEY")
    if key and key.strip().lower() in {"", "your-api-key-here"}:
        return None
    return key


def model_name() -> str:
    return get_secret("ANTHROPIC_MODEL", "claude-sonnet-5-5") or "claude-sonnet-5-5"


def timeout_seconds() -> float:
    try:
        return float(get_secret("AI_TIMEOUT_SECONDS", "30") or 30)
    except ValueError:
        return 30.0


def max_tokens() -> int:
    try:
        return int(get_secret("AI_MAX_TOKENS", "1200") or 1200)
    except ValueError:
        return 1200


def herbalist_mode() -> bool:
    """True (default): chat as a professional herbalist with graduated safety checks.
    Set HERBALIST_MODE=0 to switch back to the original strict-intake flow."""
    value = (get_secret("HERBALIST_MODE", "1") or "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def herbalist_max_tokens() -> int:
    """Consultation replies are longer than the strict-mode paraphrases."""
    try:
        return max(int(get_secret("AI_HERBALIST_MAX_TOKENS", "1800") or 1800), 600)
    except ValueError:
        return 1800


def herbalist_timeout_seconds() -> float:
    try:
        return max(float(get_secret("AI_HERBALIST_TIMEOUT_SECONDS", "60") or 60), 10.0)
    except ValueError:
        return 60.0
