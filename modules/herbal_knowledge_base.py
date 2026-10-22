"""Loads, validates and queries the local herbal knowledge base (data/herbs.json).

Design rule: missing information is never silently treated as "safe".
An entry that lacks a required safety field is marked incomplete and is
excluded from recommendations (it can still be looked up for display).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import config

# Fields every herb must have before it can be recommended.
REQUIRED_FIELDS = [
    "id", "common_name", "scientific_name", "categories", "routes", "evidence_level",
    "risks", "contraindications", "medication_interactions", "allergy_info",
    "pregnancy", "breastfeeding", "min_age_years", "sources", "last_reviewed",
]

EVIDENCE_LABELS = {
    "well_established": "Well established",
    "moderate": "Moderate evidence",
    "limited": "Limited evidence",
    "preliminary": "Preliminary research only",
    "traditional_only": "Traditional use only (no reliable clinical evidence)",
    "unknown": "Unknown",
}


@dataclass
class KnowledgeBase:
    herbs: dict[str, dict[str, Any]] = field(default_factory=dict)
    problems: dict[str, list[str]] = field(default_factory=dict)  # herb id -> missing fields

    # ----- loading -----------------------------------------------------
    @classmethod
    def load(cls, path: Path | str | None = None) -> "KnowledgeBase":
        path = Path(path or config.HERBS_PATH)
        kb = cls()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return kb  # empty KB: the app will say it has no data rather than crash
        for entry in raw.get("herbs", []):
            kb._add(entry)
        return kb

    @classmethod
    def from_entries(cls, entries: list[dict[str, Any]]) -> "KnowledgeBase":
        kb = cls()
        for entry in entries:
            kb._add(entry)
        return kb

    def _add(self, entry: dict[str, Any]) -> None:
        herb_id = str(entry.get("id") or "").strip()
        if not herb_id:
            return
        missing = [f for f in REQUIRED_FIELDS if f not in entry or entry[f] in (None, "")]
        if missing:
            self.problems[herb_id] = missing
        self.herbs[herb_id] = entry

    # ----- queries -----------------------------------------------------
    def is_complete(self, herb_id: str) -> bool:
        return herb_id in self.herbs and herb_id not in self.problems

    def get(self, herb_id: str) -> dict[str, Any] | None:
        return self.herbs.get(herb_id)

    def find_by_name(self, name: str) -> dict[str, Any] | None:
        """Return an herb whose common name, id or alias matches. None means UNKNOWN herb."""
        needle = (name or "").strip().lower()
        if len(needle) < 3:
            return None
        for herb in self.herbs.values():
            names = [herb.get("id", ""), herb.get("common_name", ""), herb.get("scientific_name", "")]
            names += list(herb.get("aliases", []))
            for n in names:
                n = str(n).lower()
                if needle == n or (len(needle) >= 4 and (needle in n or n in needle)):
                    return herb
        return None

    @staticmethod
    def routes_for(herb: dict[str, Any], category: str) -> list[str]:
        """Routes that make sense for this herb AND this concern (e.g. turmeric: oral for digestion, topical for skin)."""
        by_cat = herb.get("category_routes") or {}
        routes = by_cat.get(category, herb.get("routes", []))
        return [r for r in routes if r in herb.get("routes", [])]

    def candidates_for(self, category: str, route: str | None = None) -> list[dict[str, Any]]:
        """Complete, non-restricted herbs relevant to a concern category (and optionally route)."""
        out = []
        for herb_id, herb in self.herbs.items():
            if herb_id in self.problems or herb.get("restricted"):
                continue
            if category not in herb.get("categories", []):
                continue
            if route and route not in self.routes_for(herb, category):
                continue
            out.append(herb)
        return out

    def mentioned_in(self, text: str) -> list[dict[str, Any]]:
        """Herbs mentioned by name in free text (used to answer 'is X suitable for me?')."""
        t = (text or "").lower()
        found = []
        for herb in self.herbs.values():
            names = [herb.get("common_name", ""), *herb.get("aliases", [])]
            short = herb.get("common_name", "").split(" (")[0]
            names.append(short)
            if any(n and len(n) >= 4 and n.lower() in t for n in names):
                found.append(herb)
        return found

    def retrieval_context(self, herb_ids: list[str]) -> list[dict[str, Any]]:
        """Compact, model-safe view of selected herbs (no invented fields)."""
        keys = ["id", "common_name", "evidence_level", "routes", "risks", "traditional_uses", "evidence_supported_uses"]
        return [{k: self.herbs[h].get(k) for k in keys} for h in herb_ids if h in self.herbs]


def evidence_label(level: str | None) -> str:
    return EVIDENCE_LABELS.get(level or "unknown", EVIDENCE_LABELS["unknown"])
