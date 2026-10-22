"""Input validation helpers for user-provided text and form values."""
from __future__ import annotations

import re

MAX_USER_MESSAGE_CHARS = 1500
MAX_LIST_ITEM_CHARS = 80
MAX_LIST_ITEMS = 25

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_user_text(text: str | None) -> str:
    """Strip control characters, collapse whitespace, cap length."""
    text = _CONTROL.sub("", text or "")
    text = re.sub(r"[ \t]+", " ", text).strip()
    return text[:MAX_USER_MESSAGE_CHARS]


def validate_age(value) -> tuple[int | None, str | None]:
    """Return (age, error). Accepts ints or numeric strings."""
    if value in (None, ""):
        return None, None
    try:
        age = int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None, "Please enter age as a number."
    if age < 0 or age > 120:
        return None, "Please enter an age between 0 and 120."
    return age, None


def parse_list_field(text: str | None) -> list[str]:
    """'peanuts, lavender oil' -> ['peanuts', 'lavender oil'] with length and count limits."""
    items = []
    for part in re.split(r"[,;\n]", text or ""):
        part = _CONTROL.sub("", part).strip()
        if part and part.lower() not in {i.lower() for i in items}:
            items.append(part[:MAX_LIST_ITEM_CHARS])
    return items[:MAX_LIST_ITEMS]
