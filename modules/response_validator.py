"""Validates everything that comes back from the AI model, and re-checks recommendations.

Nothing the model returns is shown unless it passes here. Failures degrade to the
deterministic, template-based response instead of crashing or showing unvalidated text.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from modules import safety_checker as sc
from modules.herbal_knowledge_base import KnowledgeBase
from modules.product_recommender import ProductCatalogue
from modules.recommendation_engine import EngineResult
from modules.session_manager import UserProfile

MAX_MESSAGE_CHARS = 1800

# Herbs that are NOT in the knowledge base. A model must not introduce them as advice.
UNASSESSED_HERBS = [
    "kava", "st. john's wort", "st john's wort", "valerian", "ginseng", "echinacea", "garlic", "tea tree",
    "black cohosh", "saw palmetto", "ginkgo", "comfrey", "ephedra", "kratom", "cannabis", "cbd", "melatonin",
    "ginseng", "elderberry", "moringa", "shatavari", "triphala", "brahmi", "cinnamon", "clove oil", "oregano oil",
    "castor oil", "coconut oil", "apple cider vinegar", "activated charcoal", "colloidal silver", "essential oil blend",
]


@dataclass
class ValidatedResponse:
    ok: bool
    message: str = ""
    reasons: list[str] = field(default_factory=list)


def extract_json(raw: str | None) -> dict[str, Any] | None:
    """Parse a JSON object from model output (tolerates code fences and surrounding prose)."""
    if not raw or not isinstance(raw, str):
        return None
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            data = json.loads(text[start:end + 1])
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def sanitize_text(text: str) -> str:
    """Make model text safe to render: no HTML, no links/images, no control characters."""
    text = re.sub(r"<[^>]*>", "", text)                       # html tags
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)          # markdown images
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)      # markdown links -> text only
    text = re.sub(r"https?://\S+", "", text)                  # bare urls
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text.strip()


def _names_for(herb: dict[str, Any]) -> list[str]:
    names = [herb.get("common_name", "").split(" (")[0], *herb.get("aliases", [])]
    return [n.lower() for n in names if n and len(n) >= 4]


def _mentions(text: str, name: str) -> bool:
    return re.search(r"(?<![a-z])" + re.escape(name.lower()) + r"(?![a-z])", text.lower()) is not None


def validate_model_response(raw: str | None, allowed_herb_ids: set[str], kb: KnowledgeBase,
                            user_mentioned: set[str] | None = None) -> ValidatedResponse:
    """Accept only well-formed, safe, in-scope model output."""
    data = extract_json(raw)
    if data is None:
        return ValidatedResponse(False, reasons=["Model output was not valid JSON."])
    message = data.get("message")
    if not isinstance(message, str) or not message.strip():
        return ValidatedResponse(False, reasons=["Missing or empty 'message'."])
    mentioned = data.get("mentioned_herbs", [])
    if not isinstance(mentioned, list) or not all(isinstance(m, str) for m in mentioned):
        return ValidatedResponse(False, reasons=["'mentioned_herbs' must be a list of strings."])
    allowed = set(allowed_herb_ids) | set(user_mentioned or set())
    reasons: list[str] = []
    for m in mentioned:
        if m not in kb.herbs:
            reasons.append(f"Model referenced an herb that is not in the knowledge base: {m!r}.")
        elif m not in allowed:
            reasons.append(f"Model referenced an herb that was not approved: {m!r}.")
    if len(message) > MAX_MESSAGE_CHARS:
        message = message[:MAX_MESSAGE_CHARS].rsplit(" ", 1)[0] + "…"
    clean = sanitize_text(message)
    for herb_id, herb in kb.herbs.items():
        if herb_id not in allowed and any(_mentions(clean, n) for n in _names_for(herb)):
            reasons.append(f"Message mentions an herb that was not approved: {herb.get('common_name')}.")
    for name in UNASSESSED_HERBS:
        if _mentions(clean, name):
            reasons.append(f"Message mentions an unassessed ingredient: {name}.")
    reasons += [f"Unsafe wording: {r}." for r in sc.scan_output_for_violations(clean)]
    if reasons:
        return ValidatedResponse(False, reasons=reasons)
    return ValidatedResponse(True, message=clean)


@dataclass
class FinalPass:
    dropped: list[str] = field(default_factory=list)


def final_safety_pass(result: EngineResult, profile: UserProfile, kb: KnowledgeBase,
                      catalogue: ProductCatalogue) -> FinalPass:
    """Second, independent safety screen run immediately before display. Mutates `result`."""
    fp = FinalPass()
    keep = []
    for rec in result.recommendations:
        herb = kb.get(rec.herb_id or "")
        if herb is None or not kb.is_complete(herb["id"]):
            fp.dropped.append(f"{rec.name}: not in knowledge base")
            continue
        check = sc.check_herb(herb, rec.route, profile)
        if check.allowed:
            keep.append(rec)
        else:
            fp.dropped.append(f"{rec.name}: {check.blocks[0]}")
    result.recommendations = keep
    keep_p = []
    for rec in result.products:
        product = catalogue.get(rec.id)
        if product is None:
            fp.dropped.append(f"{rec.name}: not in catalogue")
            continue
        check = sc.check_product(product, kb.herbs, profile)
        if check.allowed:
            keep_p.append(rec)
        else:
            fp.dropped.append(f"{rec.name}: {check.blocks[0]}")
    result.products = keep_p
    return fp
