"""Text/format helpers shared by the conversation layer and the UI."""
from __future__ import annotations

import config
from modules.session_manager import (
    NONE, PREFER_NOT, PREG_BREASTFEEDING, PREG_NA, PREG_NEITHER, PREG_PREFER_NOT, PREG_PREGNANT,
    PREG_UNKNOWN, PROVIDED, UNSURE, UserProfile,
)

DISCLAIMER_SHORT = (
    "Herbal Wellness AI shares general educational information. It cannot diagnose conditions or replace a doctor or "
    "pharmacist. \"Natural\" does not mean safe. If you are unwell or worried, seek professional care."
)

STARTERS = [
    "Help me understand my hair fall.",
    "I want a skincare routine for sensitive skin.",
    "What herbal options may help with digestion?",
    "Help me explore nutrition for better general wellness.",
    "Check whether a herbal product is suitable for me.",
]


HERBALIST_STARTERS = [
    "I get bloated and gassy after meals.",
    "I have a sore throat and a dry cough.",
    "I can't sleep well and feel stressed.",
    "I have period cramps every month.",
    "My hair is falling out a lot.",
    "I have a bad tension headache.",
]


def herbalist_welcome_message() -> str:
    return ("Welcome, I'm your herbalist. Tell me what's troubling you, in English, Urdu or Roman Urdu, and I'll give you a practical plan "
            "straight away: what is probably behind it, which herbs and kitchen remedies suit you, how to prepare and use them, and when to see a doctor. "
            "If you take medicines or have a health condition, tell me and I'll choose around them rather than turn you away.\n\n" + DISCLAIMER_SHORT)


def category_label(cat: str | None) -> str:
    return config.CATEGORIES.get(cat or "", "general wellness")


def _status_text(status: str, items: list[str]) -> str | None:
    if status == PROVIDED and items:
        return ", ".join(items)
    if status == NONE:
        return "None reported"
    if status == UNSURE:
        return (", ".join(items) + " (and not sure about others)") if items else "Not sure"
    if status == PREFER_NOT:
        return "Prefer not to say"
    return None


def profile_rows(profile: UserProfile) -> list[tuple[str, str]]:
    """Only fields that have actually been answered. Empty fields are never shown as answered."""
    rows: list[tuple[str, str]] = []
    if profile.concern_category:
        rows.append(("Main concern", category_label(profile.concern_category)))
    if profile.concern_summary:
        rows.append(("In your words", profile.concern_summary[:120]))
    if profile.age is not None:
        rows.append(("Age group", f"{profile.age_group} ({profile.age})"))
    if profile.sex:
        rows.append(("Sex", profile.sex))
    if profile.duration_text:
        rows.append(("Duration", profile.duration_text))
    if profile.lifestyle:
        rows.append(("Lifestyle", profile.lifestyle[:120]))
    if profile.trend:
        rows.append(("Trend", profile.trend))
    if profile.severity:
        rows.append(("Severity", profile.severity))
    preg = {PREG_PREGNANT: "Pregnant", PREG_BREASTFEEDING: "Breastfeeding", PREG_NEITHER: "Not pregnant / breastfeeding",
            PREG_NA: "Not applicable", PREG_PREFER_NOT: "Prefer not to say"}.get(profile.pregnancy_status)
    if preg:
        rows.append(("Pregnancy / breastfeeding", preg))
    for label, status, items in (("Allergies", profile.allergies_status, profile.allergies),
                                 ("Medications / supplements", profile.medications_status, profile.medications),
                                 ("Medical conditions", profile.conditions_status, profile.conditions)):
        text = _status_text(status, items)
        if text:
            rows.append((label, text))
    if profile.tried_remedies:
        rows.append(("Already tried", "; ".join(profile.tried_remedies)[:150]))
    for key, label in (("route", "Preferred route"), ("diet", "Dietary preference")):
        if profile.preferences.get(key):
            rows.append((label, str(profile.preferences[key])))
    if profile.preferences.get("fragrance_free"):
        rows.append(("Product preference", "Fragrance-free"))
    return rows


def numbered(questions) -> str:
    return "\n".join(f"{i}. {q.text}" for i, q in enumerate(questions, 1))


def ack_and_questions_template(profile: UserProfile, questions, first_turn: bool) -> str:
    if first_turn and profile.concern_category:
        lead = (f"Thanks for sharing that. I can help you explore options for **{category_label(profile.concern_category).lower()}**. "
                "Before suggesting anything, I need a few details so I can keep it safe and relevant.")
    else:
        lead = "Thank you, that helps."
    return f"{lead}\n\n{numbered(questions)}\n\n_You can also answer in the Health profile in the sidebar. \"Not sure\" and \"prefer not to say\" are fine answers._"


def welcome_message() -> str:
    return ("Welcome. I work like a careful herbalist: **I take a proper patient history before I advise anything.** I will ask whether "
            "you are male or female, your age, the details of your concern, allergies, medicines, medical conditions, pregnancy where "
            "relevant, and your daily habits. The more precise your answers, the more precise and safer my guidance. Tell me what you "
            "would like help with, such as skin, hair, digestion, sleep and stress, or nutrition.\n\n" + DISCLAIMER_SHORT)


def escalation_message(flags, profile: UserProfile) -> str:
    labels = "; ".join(f.label.lower() for f in flags)
    return (f"From what you've described ({labels}), this should be looked at by a doctor or other qualified clinician **before** "
            "trying herbal remedies. I'm holding back herbal suggestions for now.\n\n"
            "Helpful things to tell them: when it started, whether it is getting worse, what you've already tried, and every medicine "
            "or supplement you use. If anything gets suddenly worse, seek urgent care.")


def no_start_message() -> str:
    return ("Let's begin with your intake. Please tell me: 1. Are you male or female? 2. How old are you? 3. What is your main concern "
            "(skin, hair, digestion, sleep and stress, nutrition, or something else), and what exactly are you noticing?")
