"""Graduated, herb-specific safety screening for the herbalist consultation.

Why this exists
---------------
The original strict engine answered "unknown" or "serious condition" with a blanket refusal for every oral
herb. A practising herbalist does not work like that. They ask: *is THIS herb a problem for THIS person?*
and, if it is, they choose a different herb, a kitchen-level amount, a topical option, or add one specific
warning. This module encodes that.

Each herb gets one verdict for one patient:

  ok       nothing relevant found
  note     worth mentioning (a minor interaction, assumed adult, "tell your pharmacist" for a concentrated product)
  caution  usable, but only with the stated precaution
  avoid    do not offer this herb to this patient (a specific, named reason is always given)

Missing information never blocks a low-risk herb. It only makes higher-risk herbs more careful, and every
unknown is turned into a question the herbalist can ask, not into a refusal.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from modules import safety_checker as sc
from modules.herbalist_kb import HerbalistKB
from modules.session_manager import (
    NONE, PREFER_NOT, PROVIDED, UNANSWERED, UNSURE, PREG_BREASTFEEDING, PREG_NEITHER, PREG_NA,
    PREG_PREFER_NOT, PREG_PREGNANT, PREG_UNKNOWN, UserProfile,
)

OK, NOTE, CAUTION, AVOID = "ok", "note", "caution", "avoid"
_RANK = {OK: 0, NOTE: 1, CAUTION: 2, AVOID: 3}


@dataclass
class HerbVerdict:
    herb_id: str
    name: str
    status: str = OK
    avoid_reasons: list[str] = field(default_factory=list)
    cautions: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)
    needs: list[str] = field(default_factory=list)   # question ids that would sharpen this verdict

    @property
    def allowed(self) -> bool:
        return self.status != AVOID

    def raise_to(self, status: str) -> None:
        if _RANK[status] > _RANK[self.status]:
            self.status = status

    def avoid(self, reason: str) -> None:
        self.raise_to(AVOID)
        if reason not in self.avoid_reasons:
            self.avoid_reasons.append(reason)

    def caution(self, text: str) -> None:
        self.raise_to(CAUTION)
        if text not in self.cautions:
            self.cautions.append(text)

    def note(self, text: str) -> None:
        self.raise_to(NOTE)
        if text not in self.notes:
            self.notes.append(text)

    def need(self, qid: str) -> None:
        if qid not in self.needs:
            self.needs.append(qid)

    def for_model(self) -> dict[str, Any]:
        out: dict[str, Any] = {"herb_id": self.herb_id, "status": self.status}
        if self.avoid_reasons:
            out["avoid_because"] = self.avoid_reasons
        if self.cautions:
            out["caution"] = self.cautions
        if self.notes:
            out["note"] = self.notes
        return out


# --------------------------------------------------------------------------- patient picture

@dataclass
class MedPicture:
    status: str = UNANSWERED
    by_class: dict[str, list[str]] = field(default_factory=dict)   # class id -> medicine names
    unclassified: list[str] = field(default_factory=list)
    no_interaction: list[str] = field(default_factory=list)

    @property
    def unknown(self) -> bool:
        """We cannot rule out an interaction we do not recognise."""
        return bool(self.unclassified) or self.status in (UNSURE, PREFER_NOT)

    @property
    def asked(self) -> bool:
        return self.status != UNANSWERED


def _fuzzy_word_in(kw: str, text: str) -> bool:
    """Whole-word match that also forgives small typos ('clacium' for 'calcium'). Short keywords must match exactly."""
    import difflib
    if sc._word_in(kw, text):
        return True
    kw = kw.lower()
    if len(kw) < 6 or " " in kw:
        return False
    return any(len(w) >= 5 and abs(len(w) - len(kw)) <= 2 and difflib.SequenceMatcher(None, w, kw).ratio() >= 0.84
               for w in re.findall(r"[a-z]+", text.lower()))


def classify_medications(profile: UserProfile, rules: dict[str, Any]) -> MedPicture:
    pic = MedPicture(status=profile.medications_status)
    classes = rules.get("med_classes", {})
    benign = rules.get("no_interaction_meds", [])
    for med in profile.medications:
        m = (med or "").strip()
        if not m:
            continue
        matched = False
        for cid, spec in classes.items():
            if any(_fuzzy_word_in(kw, m) for kw in spec.get("keywords", [])):
                pic.by_class.setdefault(cid, []).append(m)
                matched = True
        if matched:
            continue
        if any(_fuzzy_word_in(kw, m) for kw in benign):
            pic.no_interaction.append(m)
        else:
            pic.unclassified.append(m)
    return pic


def patient_condition_tags(profile: UserProfile, rules: dict[str, Any]) -> dict[str, str]:
    """Condition tags the patient has declared, or clearly mentioned in their own words (negations ignored)."""
    tags: dict[str, str] = {}
    neg = sc.load_rules().get("negation_words", [])
    declared = " ".join(profile.conditions).lower()
    free = (" ".join(profile.symptom_notes) + " " + (profile.concern_summary or "")).lower()
    for tag, kws in rules.get("condition_tags", {}).items():
        for kw in kws:
            if declared and sc._word_in(kw, declared):
                tags[tag] = "declared"
                break
            m = re.search(r"(?<![a-z0-9])" + re.escape(kw.lower()), free)
            if m and not sc._is_negated(free, m.start(), neg):
                tags.setdefault(tag, "mentioned")
                break
    return tags


def pregnancy_relevant(profile: UserProfile) -> bool:
    if profile.sex == "male":
        return False
    if profile.age is not None and (profile.age < 12 or profile.age > 55):
        return False
    return True


def preg_unknown(profile: UserProfile) -> bool:
    return pregnancy_relevant(profile) and profile.pregnancy_status in (PREG_UNKNOWN, PREG_PREFER_NOT)


# --------------------------------------------------------------------------- screening

def screen_herb(kb: HerbalistKB, herb_id: str, profile: UserProfile,
                picture: MedPicture | None = None, tags: dict[str, str] | None = None) -> HerbVerdict:
    herb = kb.get_herb(herb_id)
    if herb is None:
        v = HerbVerdict(herb_id, herb_id)
        v.note("This herb is not in my screened list, so I have not run an interaction check on it.")
        return v
    rules = kb.rules
    v = HerbVerdict(herb_id, herb.get("name", herb_id))
    risk = herb.get("risk", "moderate")
    picture = picture or classify_medications(profile, rules)
    tags = tags if tags is not None else patient_condition_tags(profile, rules)

    # ---- never-recommend list
    for hit in kb.never_recommend(herb.get("name", "") + " " + " ".join(herb.get("aka", []))):
        v.avoid(f"{hit['name']} is not something I recommend ({hit['why']}).")

    # ---- age
    min_age = herb.get("min_age") or 0
    if profile.age is None:
        if min_age >= 12:
            v.note("I'm assuming you're an adult; tell me your age to confirm.")
            v.need("age")
        v.checks.append("Age: not given, adult assumed")
    else:
        if profile.age < min_age:
            if min_age >= 18:
                v.avoid("For adults only: not recommended under 18.")
            else:
                v.avoid(f"Not recommended below about {min_age} years old.")
        elif profile.age < 18 and risk != "low":
            v.caution("Use roughly half the adult amount and check with a paediatric pharmacist.")
        elif profile.age < 12 and herb.get("child_note"):
            v.note(herb["child_note"])
        v.checks.append(f"Age: {profile.age}")

    # ---- pregnancy and breastfeeding
    status = profile.pregnancy_status
    if status in (PREG_PREGNANT, PREG_BREASTFEEDING):
        key, label = ("preg", "pregnancy") if status == PREG_PREGNANT else ("bf", "breastfeeding")
        level = herb.get(key, "limited")
        note = herb.get("preg_note" if key == "preg" else "bf_note") or ""
        if level == "avoid":
            v.avoid(f"Best avoided in {label}. {note}".strip())
        elif level == "limited":
            v.caution(f"In {label}: {note or 'only small amounts, and check with your doctor or midwife.'}")
        v.checks.append(f"Pregnancy/breastfeeding: {label} considered")
    elif preg_unknown(profile) and herb.get("preg") == "avoid":
        v.caution("Skip this if there is any chance you could be pregnant.")
        v.need("pregnancy")
    elif status in (PREG_NEITHER, PREG_NA):
        v.checks.append("Pregnancy/breastfeeding: not applicable")

    # ---- allergies
    allergy_hits: list[str] = []
    for allergy in profile.allergies:
        for kw in herb.get("allergy_kw", []):
            if sc._terms_match(allergy, kw):
                allergy_hits.append(allergy)
                break
    if allergy_hits:
        v.avoid(f"You mentioned an allergy/sensitivity ({', '.join(sorted(set(allergy_hits)))}) that can cross-react with this herb.")
    if profile.allergies_status in (NONE, PROVIDED, UNSURE, PREFER_NOT):
        v.checks.append("Allergies: checked against what you told me")
    elif risk != "low":
        v.need("allergies")

    # ---- medical conditions
    for tag in herb.get("avoid_conditions", []):
        if tag in tags:
            v.avoid(f"Not suitable with {tag} ({'you told me' if tags[tag] == 'declared' else 'you mentioned'} this).")
    for tag in herb.get("caution_conditions", []):
        if tag in tags and tag not in herb.get("avoid_conditions", []):
            v.caution(f"Because of {tag}, use only small amounts and check with your doctor.")
    serious = set(kb.rules.get("serious_conditions_graded", []))
    covered = set(herb.get("avoid_conditions", [])) | set(herb.get("caution_conditions", []))
    if risk in ("moderate", "high"):
        for tag in serious & set(tags) - covered:
            v.caution(f"Because you have {tag}, run this one past your doctor before starting.")
    if profile.conditions_status in (NONE, PROVIDED, UNSURE, PREFER_NOT) or tags:
        v.checks.append("Medical conditions: checked against what you told me")
    elif herb.get("avoid_conditions") or risk != "low":
        v.need("conditions")

    # ---- medicines
    _screen_medicines(v, herb, risk, picture)

    # ---- finishing
    if herb.get("evidence") in ("traditional_only", "preliminary"):
        v.checks.append("Evidence honesty: mainly traditional or early research")
    return v


def _screen_medicines(v: HerbVerdict, herb: dict[str, Any], risk: str, picture: MedPicture) -> None:
    interactions = herb.get("interactions", [])
    labels = {}
    for inter in interactions:
        cls = inter.get("cls")
        meds = picture.by_class.get(cls)
        if not meds:
            continue
        label = (meds[0] if len(meds) == 1 else ", ".join(meds))
        sev = inter.get("sev", "moderate")
        msg = f"Interaction with {label}: {inter.get('note', '')}".strip()
        labels[cls] = sev
        if sev == "major":
            v.avoid(msg)
        elif sev == "moderate":
            v.caution(msg + " Let your prescriber or pharmacist know.")
        else:
            v.note(msg)

    has_real_interactions = any(i.get("sev") in ("major", "moderate") for i in interactions)

    if picture.status in (NONE,) and not picture.by_class and not picture.unclassified:
        v.checks.append("Medicines: none reported")
    elif picture.asked:
        v.checks.append("Medicines: checked against what you told me")

    # Medicines I do not recognise (or "not sure" / "prefer not to say"): graded by how risky the herb is.
    if picture.unknown:
        names = ", ".join(picture.unclassified) if picture.unclassified else "your medicines"
        if risk == "high":
            v.avoid(f"I cannot confirm this is safe alongside {names}; this herb interacts with many medicines, so I'd keep it off the list until a pharmacist has checked.")
        elif risk == "moderate":
            v.caution(f"I don't have interaction data for {names}; a pharmacist can check this one quickly before you start.")
        elif has_real_interactions:
            v.note(f"I don't recognise {names}. For this herb, food and tea amounts are fine; check with a pharmacist before using concentrated supplements.")
    elif not picture.asked:
        # Medicines not asked about yet.
        if risk == "high":
            v.avoid("This herb interacts with many medicines and I do not yet know what you take.")
            v.need("medications")
        elif risk == "moderate" or has_real_interactions:
            v.need("medications")
            if risk == "moderate":
                v.note("Tell me if you take any regular medicines so I can check this one.")


def screen_many(kb: HerbalistKB, herb_ids: list[str], profile: UserProfile) -> dict[str, HerbVerdict]:
    picture = classify_medications(profile, kb.rules)
    tags = patient_condition_tags(profile, kb.rules)
    return {hid: screen_herb(kb, hid, profile, picture, tags) for hid in herb_ids}


def summarize_profile_gaps(profile: UserProfile) -> list[str]:
    """What the herbalist still does not know (used as a hint to ask, never as a reason to refuse)."""
    gaps = []
    if profile.age is None:
        gaps.append("age")
    if preg_unknown(profile):
        gaps.append("pregnancy_or_breastfeeding")
    if profile.medications_status == UNANSWERED:
        gaps.append("medicines")
    if profile.conditions_status == UNANSWERED:
        gaps.append("medical_conditions")
    if profile.allergies_status == UNANSWERED:
        gaps.append("allergies")
    return gaps
