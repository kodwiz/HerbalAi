"""Loads and searches the herbalist knowledge base (herbs, minor-condition protocols, safety rules).

This is the "textbook" the consultation engine and the AI herbalist read from. It is separate from the
legacy strict knowledge base (herbal_knowledge_base.py / data/herbs.json) so that both can coexist.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import config

REQUIRED_HERB_FIELDS = ["name", "risk", "evidence", "prep", "preg", "bf", "min_age", "interactions", "allergy_kw"]
REQUIRED_CONDITION_FIELDS = ["title", "category", "keywords", "first_line", "second_line", "self_care", "see_doctor_if"]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\u2019", "'").lower().strip())


def _kw_pattern(kw: str) -> re.Pattern[str]:
    """Keyword -> regex. 'cold$' means the whole word; otherwise it matches the start of a word ('bloat' -> 'bloating')."""
    kw = kw.lower()
    if kw.endswith("$"):
        return re.compile(r"(?<![a-z0-9])" + re.escape(kw[:-1]) + r"(?![a-z0-9])")
    return re.compile(r"(?<![a-z0-9])" + re.escape(kw))


@dataclass
class ConditionMatch:
    condition_id: str
    score: float
    matched: list[str] = field(default_factory=list)


@dataclass
class HerbalistKB:
    herbs: dict[str, dict[str, Any]] = field(default_factory=dict)
    conditions: dict[str, dict[str, Any]] = field(default_factory=dict)
    rules: dict[str, Any] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    _kw: dict[str, list[tuple[str, re.Pattern[str]]]] = field(default_factory=dict, repr=False)
    _herb_names: list[tuple[str, re.Pattern[str]]] = field(default_factory=list, repr=False)

    # ------------------------------------------------------------ loading
    @classmethod
    def load(cls, herbs_path: Path | str | None = None, conditions_path: Path | str | None = None,
             rules_path: Path | str | None = None) -> "HerbalistKB":
        kb = cls()
        kb.herbs = _read_json(herbs_path or config.HERBALIST_HERBS_PATH, kb).get("herbs", {})
        kb.conditions = _read_json(conditions_path or config.HERBALIST_CONDITIONS_PATH, kb).get("conditions", {})
        kb.rules = _read_json(rules_path or config.HERBALIST_RULES_PATH, kb)
        kb._validate()
        kb._index()
        return kb

    def _validate(self) -> None:
        for hid, herb in self.herbs.items():
            missing = [f for f in REQUIRED_HERB_FIELDS if f not in herb]
            if missing:
                self.problems.append(f"herb {hid}: missing {', '.join(missing)}")
        for cid, cond in self.conditions.items():
            missing = [f for f in REQUIRED_CONDITION_FIELDS if f not in cond]
            if missing:
                self.problems.append(f"condition {cid}: missing {', '.join(missing)}")
            for tier in ("first_line", "second_line"):
                for entry in cond.get(tier, []):
                    herb = self.herbs.get(entry.get("herb", ""))
                    if herb is None:
                        self.problems.append(f"condition {cid}: unknown herb {entry.get('herb')!r}")
                    elif entry.get("prep", 0) >= len(herb.get("prep", [])):
                        self.problems.append(f"condition {cid}: {entry['herb']} prep index {entry.get('prep')} out of range")

    def _index(self) -> None:
        self._kw = {cid: [(kw, _kw_pattern(kw)) for kw in cond.get("keywords", [])] for cid, cond in self.conditions.items()}
        names: list[tuple[str, re.Pattern[str]]] = []
        for hid, herb in self.herbs.items():
            variants = {hid.replace("_", " "), herb.get("name", "").split(" (")[0], *herb.get("aka", [])}
            for v in variants:
                v = _norm(v)
                if len(v) >= 4:
                    names.append((hid, re.compile(r"(?<![a-z0-9])" + re.escape(v) + r"(?![a-z0-9])")))
        self._herb_names = names

    @property
    def loaded(self) -> bool:
        return bool(self.herbs and self.conditions)

    # ------------------------------------------------------------ herb lookup
    def get_herb(self, herb_id: str) -> dict[str, Any] | None:
        return self.herbs.get(herb_id)

    def herbs_mentioned(self, text: str) -> list[str]:
        """Ids of herbs named in free text (names, aliases and common local names)."""
        t = _norm(text)
        found: list[str] = []
        for hid, pat in self._herb_names:
            if hid not in found and pat.search(t):
                found.append(hid)
        return found

    def find_herb(self, name: str) -> str | None:
        """Resolve a name the model or patient used to a herb id; None if the herb is not in the knowledge base."""
        n = _norm(name)
        if not n:
            return None
        if n.replace(" ", "_") in self.herbs:
            return n.replace(" ", "_")
        hits = self.herbs_mentioned(n)
        return hits[0] if hits else None

    # ------------------------------------------------------------ condition lookup
    def match_conditions(self, text: str, limit: int = 2) -> list[ConditionMatch]:
        """Rank condition protocols by how well their keywords match the text. Specific phrases score higher."""
        t = _norm(text)
        scored: list[ConditionMatch] = []
        for cid, patterns in self._kw.items():
            score, matched = 0.0, []
            for kw, pat in patterns:
                if pat.search(t):
                    words = len(re.findall(r"[a-z0-9']+", kw))
                    score += 1.0 + 0.5 * (words - 1)
                    matched.append(kw)
            if score > 0:
                scored.append(ConditionMatch(cid, score, matched))
        scored.sort(key=lambda m: (-m.score, m.condition_id))
        if not scored:
            return []
        top = scored[0].score
        return [m for m in scored[:limit] if m.score >= 0.7 * top]

    def condition(self, condition_id: str) -> dict[str, Any] | None:
        return self.conditions.get(condition_id)

    def topic_titles(self) -> list[str]:
        return [c["title"] for c in self.conditions.values()]

    # ------------------------------------------------------------ rules
    def never_recommend(self, text: str) -> list[dict[str, str]]:
        """Entries from the never-recommend list that appear in the text."""
        t = _norm(text)
        hits = []
        for entry in self.rules.get("never_recommend", []):
            names = [entry["name"].split(" (")[0], *entry.get("aliases", [])]
            for n in names:
                n = _norm(n)
                if len(n) >= 4 and re.search(r"(?<![a-z0-9])" + re.escape(n) + r"(?![a-z0-9])", t):
                    hits.append(entry)
                    break
        return hits


def _read_json(path: Path | str, kb: HerbalistKB) -> dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        kb.problems.append(f"could not load {Path(path).name}: {exc}")
        return {}
