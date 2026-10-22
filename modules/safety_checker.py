"""Deterministic safety engine. No AI model is involved in any decision here.

Principles
* Missing data is never treated as safe.
* A block can only be removed by changing the profile/rules, never by the model.
* Every function is pure and independently testable.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import config
from modules.session_manager import (
    ANSWERED_STATUSES, PREG_BREASTFEEDING, PREG_NA, PREG_NEITHER, PREG_PREFER_NOT,
    PREG_PREGNANT, PREG_UNKNOWN, UserProfile,
)

# ---------------------------------------------------------------- data classes


@dataclass
class EmergencyHit:
    rule_id: str
    label: str
    matched_text: str
    is_crisis: bool = False


@dataclass
class RedFlag:
    rule_id: str
    label: str
    action: str  # "block_herbal" | "advise_clinician"


@dataclass
class SafetyResult:
    """Outcome of screening ONE herb/product route for ONE user."""
    allowed: bool = True
    blocks: list[str] = field(default_factory=list)       # reasons the item must not be recommended
    cautions: list[str] = field(default_factory=list)     # must be shown with the recommendation
    checks: list[str] = field(default_factory=list)       # human-readable record of checks performed
    interaction_status: str = "none_found"                 # none_found | flagged | insufficient
    missing_info: list[str] = field(default_factory=list)  # ids of questions that would resolve a block

    def block(self, reason: str) -> None:
        self.allowed = False
        if reason not in self.blocks:
            self.blocks.append(reason)

    def caution(self, text: str) -> None:
        if text not in self.cautions:
            self.cautions.append(text)


# ---------------------------------------------------------------- rule loading

@lru_cache(maxsize=1)
def _load_rules_cached(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_rules(path: str | None = None) -> dict[str, Any]:
    return _load_rules_cached(str(path or config.SAFETY_RULES_PATH))


# ---------------------------------------------------------------- text helpers

def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower().strip())


def _word_in(needle: str, haystack: str) -> bool:
    needle, haystack = _norm(needle), _norm(haystack)
    if len(needle) < 3:
        return False
    return re.search(r"(?<![a-z0-9])" + re.escape(needle) + r"(?![a-z0-9])", haystack) is not None


def _terms_match(a: str, b: str) -> bool:
    return _word_in(a, b) or _word_in(b, a)


def _is_negated(text: str, start: int, negations: list[str]) -> bool:
    """True if the match at `start` is negated within the same clause ("no chest pain")."""
    before = text[max(0, start - 45):start]
    # only look inside the current clause
    clause = re.split(r"[.;,!?]|\bbut\b|\bhowever\b|\balthough\b", before)[-1]
    clause = " " + _norm(clause) + " "
    return any(f" {n} " in clause for n in negations)


# ---------------------------------------------------------------- emergencies

def scan_for_emergency(text: str, rules: dict[str, Any] | None = None) -> list[EmergencyHit]:
    """Detect emergency warning signs in free text. Negated mentions are ignored."""
    rules = rules or load_rules()
    t = (text or "").lower()
    negs = rules.get("negation_words", [])
    hits: list[EmergencyHit] = []
    for rule in rules["emergency_rules"]:
        for pat in rule["patterns"]:
            for m in re.finditer(pat, t, flags=re.IGNORECASE):
                if _is_negated(t, m.start(), negs):
                    continue
                hits.append(EmergencyHit(rule["id"], rule["label"], m.group(0),
                                         is_crisis=rule.get("message_type") == "crisis"))
                break
            else:
                continue
            break
    return hits


EMERGENCY_MESSAGE = (
    "**This could be an emergency.** Please contact your local emergency number or go to the nearest "
    "emergency department right now. Do not wait to try herbs, teas or home remedies. If you are "
    "alone, ask someone nearby to help or to stay with you."
)
CRISIS_MESSAGE = (
    "I'm really glad you said something. You deserve support right now. Please contact your local "
    "emergency number or a crisis line in your country, or reach out to someone you trust and tell "
    "them how you are feeling. If you might act on these thoughts, call emergency services now. "
    "I'm not able to offer herbal advice in this situation, but I can stay in this conversation with you."
)


def emergency_response(hits: list[EmergencyHit]) -> str:
    if any(h.is_crisis for h in hits):
        return CRISIS_MESSAGE
    labels = ", ".join(sorted({h.label for h in hits}))
    return f"{EMERGENCY_MESSAGE}\n\nWhat I noticed: {labels}."


# ---------------------------------------------------------------- red flags

def check_red_flags(profile: UserProfile, extra_text: str = "",
                    rules: dict[str, Any] | None = None) -> list[RedFlag]:
    """Concern-specific warning rules (not one generic checklist)."""
    rules = rules or load_rules()
    category = profile.concern_category or "general"
    text = _norm(profile.all_symptom_text() + " " + extra_text)
    negs = rules.get("negation_words", [])
    found: list[RedFlag] = []
    cats = [category] if category == "general" else [category, "general"]
    for cat in cats:
        for rule in rules["concern_red_flags"].get(cat, []):
            triggered = False
            for pat in rule.get("patterns", []):
                for m in re.finditer(pat, text, flags=re.IGNORECASE):
                    if not _is_negated(text, m.start(), negs):
                        triggered = True
                        break
                if triggered:
                    break
            limit = rule.get("duration_days_over")
            if limit and profile.duration_days is not None and profile.duration_days > limit:
                triggered = True
            if triggered and all(f.rule_id != rule["id"] for f in found):
                found.append(RedFlag(rule["id"], rule["label"], rule["action"]))
    return found


# ---------------------------------------------------------------- allergies

def expand_allergy_terms(allergies: list[str], rules: dict[str, Any] | None = None) -> list[str]:
    """User allergy phrases plus related ingredient words (e.g. 'tree nuts' -> shea, almond...)."""
    rules = rules or load_rules()
    syn = rules.get("allergen_synonyms", {})
    terms: list[str] = []
    for a in allergies:
        a_n = _norm(a)
        if not a_n:
            continue
        terms.append(a_n)
        for key, extras in syn.items():
            if re.search(r"(?<![a-z])" + re.escape(key) + r"s?(?![a-z])", a_n):
                terms.extend(extras)
    return sorted(set(terms))


def herb_allergen_keywords(herb: dict[str, Any]) -> list[str]:
    kws = list(herb.get("allergy_info", {}).get("keywords", []))
    kws += [herb.get("common_name", "").split(" (")[0], *herb.get("aliases", [])]
    sci = herb.get("scientific_name", "")
    if sci:
        kws.append(sci.split()[0])
    return [k.lower() for k in kws if k]


def allergy_conflicts_with_herb(allergies: list[str], herb: dict[str, Any],
                                rules: dict[str, Any] | None = None) -> list[str]:
    terms = expand_allergy_terms(allergies, rules)
    kws = herb_allergen_keywords(herb)
    hits = [t for t in terms if any(_terms_match(t, k) for k in kws)]
    return sorted(set(hits))


def allergy_conflicts_with_ingredients(allergies: list[str], ingredients: list[str],
                                       rules: dict[str, Any] | None = None) -> list[str]:
    terms = expand_allergy_terms(allergies, rules)
    hits = [t for t in terms if any(_terms_match(t, ing) for ing in ingredients)]
    return sorted(set(hits))


# ---------------------------------------------------------------- conditions

def user_condition_tags(profile: UserProfile, rules: dict[str, Any] | None = None,
                        include_symptoms: bool = True) -> set[str]:
    rules = rules or load_rules()
    text = _norm(" ".join(profile.conditions))
    if include_symptoms:
        text += " " + _norm(profile.all_symptom_text())
    tags = set()
    for tag, kws in rules["condition_tags"].items():
        if any(_word_in(k, text) or _norm(k) in text for k in kws):
            tags.add(tag)
    return tags


def contraindicated_conditions(herb: dict[str, Any], profile: UserProfile,
                               rules: dict[str, Any] | None = None) -> list[str]:
    rules = rules or load_rules()
    tags = user_condition_tags(profile, rules)
    text = _norm(" ".join(profile.conditions))
    hits = []
    for cond in herb.get("contraindications", {}).get("conditions", []):
        c = _norm(cond)
        if c in tags or (c and c in text):
            hits.append(cond)
    return hits


# ---------------------------------------------------------------- medications

def classify_medication(med: str, rules: dict[str, Any] | None = None) -> tuple[set[str], str]:
    """Return (classes, status). status: 'classified' | 'recognized_no_data' | 'unknown'."""
    rules = rules or load_rules()
    m = _norm(med)
    classes = {cls for cls, kws in rules["medication_classes"].items() if any(_word_in(k, m) for k in kws)}
    if classes:
        return classes, "classified"
    if any(_word_in(k, m) for k in rules.get("other_known_medications", [])):
        return set(), "recognized_no_data"
    return set(), "unknown"


def check_interactions(herb: dict[str, Any], route: str, profile: UserProfile,
                       result: SafetyResult, rules: dict[str, Any] | None = None) -> None:
    rules = rules or load_rules()
    meds = [m for m in profile.medications if m.strip()]
    if not meds:
        if profile.medications_status in ANSWERED_STATUSES:
            result.checks.append("Medication check: none reported.")
        return
    flagged = False
    unknown: list[str] = []
    for med in meds:
        classes, status = classify_medication(med, rules)
        if status == "unknown":
            unknown.append(med)
            continue
        if status == "recognized_no_data":
            result.caution(f"No interaction data for '{med}' in this app's limited dataset. Confirm with a pharmacist before use.")
            continue
        for inter in herb.get("medication_interactions", []):
            cls = inter.get("medication_class")
            if cls == "any_oral_medication":
                continue
            if cls in classes:
                sev = inter.get("severity", "moderate")
                msg = f"Possible interaction with '{med}' ({cls.replace('_', ' ')}): {inter.get('note', '')}"
                flagged = True
                if sev in ("major", "moderate") and not (route == "topical" and sev == "moderate"):
                    result.block(msg + " Speak with a pharmacist or doctor first.")
                else:
                    result.caution(msg)
    if route == "oral":
        for inter in herb.get("medication_interactions", []):
            if inter.get("medication_class") == "any_oral_medication":
                result.caution("Fibre/bulk products can change absorption of other medicines: " + inter.get("note", "") )
                flagged = True
    if unknown:
        result.interaction_status = "insufficient"
        names = ", ".join(unknown)
        if route == "oral":
            result.block(f"Interaction information insufficient for: {names}. A pharmacist or doctor should check this first.")
        else:
            result.caution(f"Interaction information insufficient for: {names}. Ask a pharmacist if unsure.")
    elif flagged:
        result.interaction_status = "flagged"
    result.checks.append("Medication check completed against this app's limited interaction dataset (not a complete interaction check).")


# ---------------------------------------------------------------- required info

def missing_safety_info(profile: UserProfile, route: str) -> list[str]:
    """Question ids whose answers are required before a personalised recommendation."""
    missing = []
    if profile.sex is None:
        missing.append("sex")
    if profile.age is None:
        missing.append("age")
    if profile.allergies_status not in ANSWERED_STATUSES:
        missing.append("allergies")
    if _pregnancy_relevant(profile) and profile.pregnancy_status == PREG_UNKNOWN:
        missing.append("pregnancy")
    if route == "oral":
        if profile.medications_status not in ANSWERED_STATUSES:
            missing.append("medications")
        if profile.conditions_status not in ANSWERED_STATUSES:
            missing.append("conditions")
    return missing


def _pregnancy_relevant(profile: UserProfile) -> bool:
    if profile.sex == "male":
        return False
    if profile.age is not None and (profile.age < 12 or profile.age > 55):
        return False
    return True


# ---------------------------------------------------------------- main herb check

def check_herb(herb: dict[str, Any], route: str, profile: UserProfile,
               rules: dict[str, Any] | None = None) -> SafetyResult:
    """Screen one herb for one route. Safety always wins over relevance."""
    rules = rules or load_rules()
    r = SafetyResult()
    hr = rules.get("high_risk_rules", {})

    if route not in herb.get("routes", []):
        r.block(f"{herb.get('common_name')} is not assessed by this app for {route} use.")
        return r
    if herb.get("restricted"):
        r.block(f"{herb.get('common_name')} is a higher-risk herb that this app does not recommend without clinician review.")

    # sex (strict intake: some herbs and risks differ by sex and hormonal status)
    if profile.sex is None:
        r.block("Sex is not known yet, so sex- and hormone-related precautions cannot be checked.")
        r.missing_info.append("sex")

    # age
    min_ages = herb.get("min_age_years") or {}
    min_age = min_ages.get(route, config.DEFAULT_MIN_AGE_YEARS) if route in min_ages else config.DEFAULT_MIN_AGE_YEARS
    if profile.age is None:
        r.block("Age not known yet, so age restrictions cannot be checked.")
        r.missing_info.append("age")
    elif profile.age <= hr.get("child_max_age_years", 17) or profile.age < (min_age or config.DEFAULT_MIN_AGE_YEARS):
        r.block("This app does not give herbal recommendations for children or adolescents. Please ask a paediatrician or pharmacist.")
    r.checks.append("Age restriction check")

    # pregnancy / breastfeeding
    if profile.is_pregnant_or_breastfeeding:
        key = "pregnancy" if profile.pregnancy_status == PREG_PREGNANT else "breastfeeding"
        status = (herb.get(key) or {}).get(route, "insufficient_data")
        if status != "explicitly_ok":
            word = "pregnancy" if key == "pregnancy" else "breastfeeding"
            r.block(f"Safety during {word} is {'not recommended' if status == 'avoid' else 'not established'} for {herb.get('common_name')} ({route}). Please ask your doctor or midwife.")
        r.checks.append("Pregnancy/breastfeeding check")
    elif _pregnancy_relevant(profile) and profile.pregnancy_status in (PREG_UNKNOWN, PREG_PREFER_NOT):
        r.block("Pregnancy/breastfeeding status is unknown, so safety cannot be confirmed.")
        r.missing_info.append("pregnancy")

    # allergies
    if profile.allergies_status in ANSWERED_STATUSES:
        conflicts = allergy_conflicts_with_herb(profile.allergies, herb, rules)
        if conflicts:
            r.block(f"Allergy/sensitivity conflict ({', '.join(conflicts)}) with {herb.get('common_name')}"
                    f" ({herb.get('allergy_info', {}).get('family', 'plant family')}).")
        r.checks.append("Allergy check")
    else:
        r.missing_info.append("allergies")
        if route == "oral":
            r.block("Allergy status is unknown or not shared, so oral herbs cannot be screened.")
        else:
            r.caution("Your allergy status is unknown. Do a patch test first and stop at any sign of reaction.")

    # conditions
    hits = contraindicated_conditions(herb, profile, rules)
    if hits:
        r.block(f"Listed as unsuitable with: {', '.join(hits)}.")
    serious = set(hr.get("serious_conditions_requiring_clinician", []))
    if route == "oral" and (user_condition_tags(profile, rules, include_symptoms=False) & serious):
        r.block("You mentioned a significant medical condition. Oral herbs should be cleared by your clinician first.")
    if route == "oral" and profile.conditions_status not in ANSWERED_STATUSES:
        r.block("Medical background is unknown or not shared, so oral herbs cannot be screened.")
        if "conditions" not in r.missing_info:
            r.missing_info.append("conditions")
    r.checks.append("Medical-condition check")

    # medications
    if route == "oral" and profile.medications_status not in ANSWERED_STATUSES:
        r.block("Medication use is unknown or not shared, so interactions cannot be ruled out.")
        r.interaction_status = "insufficient"
        r.missing_info.append("medications")
    else:
        check_interactions(herb, route, profile, r, rules)

    # evidence honesty
    if herb.get("evidence_level") in ("traditional_only", "preliminary", "limited"):
        r.caution("Evidence is limited: this has been used traditionally or studied only in small trials. It is not proven to work.")
    # review status
    if herb.get("review_status") == "draft_requires_clinical_review":
        r.caution("This knowledge-base entry is a draft that has not yet been clinically reviewed.")
    for line in herb.get("risks", [])[:2]:
        r.caution(line)
    r.missing_info = sorted(set(r.missing_info))
    return r


def check_product(product: dict[str, Any], herbs_by_id: dict[str, dict[str, Any]], profile: UserProfile,
                  rules: dict[str, Any] | None = None) -> SafetyResult:
    """Screen a catalogue product: every ingredient, plus the herb(s) it is based on."""
    rules = rules or load_rules()
    route = product.get("route", "oral")
    result = SafetyResult()
    herb_ids = product.get("herb_ids") or []
    if not herb_ids:
        result.block("Product is not linked to any assessed herb, so it cannot be screened.")
        return result
    for hid in herb_ids:
        herb = herbs_by_id.get(hid)
        if herb is None:
            result.block(f"Ingredient '{hid}' is not in the knowledge base (unknown), so safety cannot be assessed.")
            continue
        sub = check_herb(herb, route, profile, rules)
        for b in sub.blocks:
            result.block(b)
        for c in sub.cautions:
            result.caution(c)
        result.missing_info += sub.missing_info
        result.checks += sub.checks
        if sub.interaction_status != "none_found":
            result.interaction_status = sub.interaction_status
    if profile.allergies_status in ANSWERED_STATUSES:
        hits = allergy_conflicts_with_ingredients(profile.allergies, product.get("ingredients", []), rules)
        if hits:
            result.block(f"Product ingredient list conflicts with a reported allergy/sensitivity ({', '.join(hits)}).")
        result.checks.append("Full ingredient-list allergy check")
    if not product.get("ingredients"):
        result.block("Product has no verified ingredient list.")
    if product.get("is_demo"):
        result.caution("Demonstration entry only: not a real product. Do not purchase or use based on this listing.")
    result.missing_info = sorted(set(result.missing_info))
    return result


# ---------------------------------------------------------------- final output guard

DANGEROUS_PATTERNS = [
    (r"\b(stop|quit|discontinue|skip)\b[^.]{0,40}\b(your|the|prescribed)\b[^.]{0,30}\b(medication|medicine|pills|tablets|insulin|treatment)\b", "tells user to stop medication"),
    (r"\bcure[sd]?\b", "claims a cure"),
    (r"\bguarantee[sd]?\b", "guarantees an outcome"),
    (r"\b100\s*%\s*(safe|effective)\b", "absolute safety/efficacy claim"),
    (r"\bno\s+side\s+effects\b", "denies side effects"),
    (r"\b(you\s+have|you\s+are\s+suffering\s+from|this\s+is\s+definitely|you\s+probably\s+have)\b\s+(a\s+|an\s+)?(thyroid|diabet|cancer|alopecia|eczema|psoriasis|ibs|crohn|pcos|anemia|anaemia|infection)", "appears to diagnose"),
    (r"\b(instead\s+of|rather\s+than|no\s+need\s+(for|to\s+see))\b[^.]{0,30}\b(doctor|physician|clinician|medical)\b", "discourages medical care"),
    (r"\b(natural(ly)?\s+means\s+safe|because\s+it(\s+is|'s)\s+natural,?\s+it(\s+is|'s)\s+safe)\b", "equates natural with safe"),
    (r"\b\d+(\.\d+)?\s*(mg|g|grams?|ml|mcg|tsp|tbsp|teaspoons?|tablespoons?|capsules?)\b\s*(per|a|each|/)\s*(day|daily|dose)", "states a dose"),
    (r"\b(take|use|consume|drink|swallow)\b[^.]{0,25}\b\d+(\.\d+)?\s*(mg|g|ml|mcg|tsp|tbsp|teaspoons?|tablespoons?|capsules?)\b", "states a dose"),
]


_NEGATION_BEFORE = re.compile(r"(\bnot|n't|\bnever|\bno|\bcannot|\bwithout|\bnor)\W+(\w+\W+){0,3}$", re.IGNORECASE)
_NEGATABLE = {"claims a cure", "guarantees an outcome"}


def scan_output_for_violations(text: str) -> list[str]:
    """Final guard on any text shown to the user. Returns reasons the text must not be shown.

    "does not cure" / "no guarantee" are fine; claims of a cure or guarantee are not.
    """
    problems = []
    text = text or ""
    for pat, why in DANGEROUS_PATTERNS:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            if why in _NEGATABLE and _NEGATION_BEFORE.search(text[max(0, m.start() - 40):m.start()]):
                continue
            problems.append(why)
            break
    return problems
