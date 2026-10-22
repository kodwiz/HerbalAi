"""Professional-herbalist consultation.

One chat turn flows like this:

  emergency scan (conversation_manager)        -> stop and send to emergency care
  profile update from the patient's words
  match the concern to a minor-condition protocol (and herbs the patient names)
  triage: urgent / see a clinician soon / self-care     (warning signs from the patient's own words)
  AI herbalist writes the consultation, grounded in the protocol and the herb monographs
      -> guardrails check it: no hard-contraindicated herb, no banned herb, no dangerous wording,
         triage consistent with warning signs; one corrective retry
  if the AI is unavailable or fails the guardrails -> the built-in herbalist composes the plan from the same
  knowledge base (so the app is a working herbalist even with no API key)

Nothing here replaces a blanket refusal with a blanket yes: every herb is screened for THIS patient
(herbalist_screening.py) and the answer to a problem is always a safer alternative, never a dead end.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from modules import health_assessment as ha
from modules import safety_checker as sc
from modules.ai_client import AIClient, AIError
from modules.herbal_knowledge_base import evidence_label
from modules.herbalist_kb import HerbalistKB
from modules.herbalist_screening import (
    AVOID, CAUTION, HerbVerdict, classify_medications, patient_condition_tags, pregnancy_relevant,
    screen_herb, screen_many, summarize_profile_gaps,
)
from modules.recommendation_engine import Recommendation
from modules.response_validator import extract_json, sanitize_text
from modules.session_manager import (
    NONE, PREG_BREASTFEEDING, PREG_NEITHER, PREG_PREGNANT, PROVIDED, SessionState, UserProfile,
)

SELF_CARE, SEE_CLINICIAN, URGENT = "self_care", "see_clinician_soon", "urgent"
MAX_ITEMS = 3
HISTORY_TURNS = 12
MAX_MESSAGE_CHARS = 4800

_THANKS = re.compile(r"^\s*(thanks|thank you|thankyou|thx|ok thanks|okay thanks|shukriya|shukria|jazak|great thanks|got it|ok$|okay$)\W*$", re.I)
_GREETING = re.compile(r"^\s*(hi|hello|hey|salam|salaam|assalam|assalamu|aoa|good (morning|evening|afternoon)|namaste)\b", re.I)
_SUITABILITY = re.compile(r"\b(safe|suitable|okay|ok to|ok for|can i (use|take|have|drink)|should i (use|take|drink)|is it fine|side effects?|interact)\b", re.I)


@lru_cache(maxsize=1)
def default_kb() -> HerbalistKB:
    return HerbalistKB.load()


# ============================================================================ small helpers

def _msg(content: str, **extra: Any) -> dict[str, Any]:
    msg = {"role": "assistant", "content": content, "recommendations": [], "products": [], "withheld": [],
           "notices": [], "lifestyle": [], "mode": "guided", "plan_heading": "Details for the herbs above",
           "cards_collapsed": True}
    msg.update(extra)
    return msg


def _short_name(herb: dict[str, Any]) -> str:
    return herb.get("name", "").split(" (")[0]


def _recent_user_texts(session: SessionState, n: int = 4) -> list[str]:
    return [m["content"] for m in session.messages if m["role"] == "user"][-n:]


# ============================================================================ triage

@dataclass
class TriageInfo:
    level: str = SELF_CARE
    labels: list[str] = field(default_factory=list)         # what was noticed
    advise_check: list[str] = field(default_factory=list)   # non-blocking "also get this checked" notes
    urgent_label: str | None = None


# Self-limiting problems that should be examined when they last longer than this many days.
DURATION_LIMIT_DAYS = {
    "cough": 21, "sore_throat": 14, "common_cold": 14, "nasal_congestion_sinus": 21, "diarrhea_mild": 7, "nausea": 7,
    "fever_mild": 3, "indigestion_bloating": 28, "heartburn_reflux": 42, "tension_headache": 21, "insomnia_mild": 42,
    "mouth_ulcers_gum": 21, "cold_sores": 14, "minor_wounds": 14, "insect_bites": 7, "burns_sunburn": 14,
}
_GLOBAL_FLAGS = [
    (r"(losing|lost|loss of)\s+(a\s+lot\s+of\s+)?weight(\s+(without|unintentionally|unexplained|suddenly))?|weight\s+loss|(dropped|losing)\s+\d+\s*(kg|kilos?|pounds|lbs)", "unexplained weight loss"),
    (r"night\s+sweats", "night sweats"),
    (r"(blood\s+in\s+(my\s+)?(urine|stool|poo|motion|vomit))|(urine|stool)\s+(has|with)\s+blood", "blood in urine, stool or vomit"),
    (r"lump|swelling\s+that\s+(won'?t|does\s*not)\s+go", "a new lump or persistent swelling"),
]
_DUR = re.compile(r"(\d+|a|one|two|three|four|five|six|several)\s*(day|week|month|year)s?", re.I)
_NUMWORDS = {"a": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "several": 3}


_NOT_DURATION = re.compile(
    r"\d+\s*(?:weeks?|months?)\s*(?:\(?pregnant\)?|along|gestation)|(?:pregnant|pregnancy)\s*(?:for|since|at)?\s*\d+\s*(?:weeks?|months?)|"
    r"\d+\s*(?:years?|yrs?)[\s-]*old|(?:baby|child|son|daughter|infant)\s+is\s+\d+\s*(?:days?|weeks?|months?)|\d+\s*(?:weeks?|months?)\s*(?:postpartum|after (?:birth|delivery))", re.I)


def longest_duration_days(text: str) -> int:
    best = 0
    text = _NOT_DURATION.sub(" ", text)
    for m in _DUR.finditer(text):
        n = m.group(1).lower()
        n = int(n) if n.isdigit() else _NUMWORDS.get(n, 1)
        best = max(best, n * {"day": 1, "week": 7, "month": 30, "year": 365}[m.group(2).lower()])
    return best


def assess_triage(session: SessionState, kb: HerbalistKB, condition_ids: list[str], recent_text: str) -> TriageInfo:
    """Warning signs from the patient's own words. Negated mentions ('no chest pain') are ignored."""
    info = TriageInfo()
    text = recent_text.lower()
    neg = sc.load_rules().get("negation_words", [])

    # 1. condition-specific red flags
    for cid in condition_ids:
        cond = kb.condition(cid) or {}
        for rf in cond.get("red_flags", []):
            for m in re.finditer(rf["pattern"], text, flags=re.I):
                if sc._is_negated(text, m.start(), neg):
                    continue
                if rf["level"] == "urgent":
                    info.level, info.urgent_label = URGENT, rf["label"]
                elif info.level != URGENT:
                    info.level = SEE_CLINICIAN
                if rf["label"] not in info.labels:
                    info.labels.append(rf["label"])
                break
        if cond.get("triage") == "see_clinician" and info.level == SELF_CARE:
            info.level = SEE_CLINICIAN
            info.labels.append(cond["title"].lower() + " usually needs a proper check")

    # 1b. global warning signs and prolonged duration
    neg_text = text
    for pat, label in _GLOBAL_FLAGS:
        for m in re.finditer(pat, text, flags=re.I):
            if sc._is_negated(neg_text, m.start(), neg):
                continue
            if info.level != URGENT:
                info.level = SEE_CLINICIAN
            if label not in info.labels:
                info.labels.append(label)
            break
    days = longest_duration_days(text)
    for cid in condition_ids:
        limit = DURATION_LIMIT_DAYS.get(cid)
        if limit and days > limit:
            if info.level != URGENT:
                info.level = SEE_CLINICIAN
            label = f"{(kb.condition(cid) or {}).get('title', cid).lower()} lasting longer than {limit // 7 if limit >= 14 else limit} {'weeks' if limit >= 14 else 'days'}"
            if label not in info.labels:
                info.labels.append(label)

    # 2. original concern-category rules (skin infection, GI bleeding, hair patchiness, mood flags ...)
    menopause = "menopause_symptoms" in condition_ids
    for flag in sc.check_red_flags(session.profile):
        if menopause and flag.rule_id in ("general_alarm", "systemic_illness"):
            continue  # night sweats/flushes are expected in menopause
        if flag.action == "block_herbal":
            if info.level != URGENT:
                info.level = SEE_CLINICIAN
            if flag.label not in info.labels:
                info.labels.append(flag.label.lower())
        elif flag.label not in info.advise_check:
            info.advise_check.append(flag.label.lower())
    return info


# ============================================================================ plan (built-in herbalist)

@dataclass
class PlanItem:
    herb_id: str
    prep_idx: int
    why: str
    verdict: HerbVerdict
    tier: int


@dataclass
class Plan:
    condition_id: str
    title: str
    items: list[PlanItem] = field(default_factory=list)
    withheld: list[tuple[str, list[str]]] = field(default_factory=list)


def build_plan(kb: HerbalistKB, profile: UserProfile, condition_id: str, max_items: int = MAX_ITEMS) -> Plan:
    cond = kb.condition(condition_id) or {}
    plan = Plan(condition_id, cond.get("title", condition_id))
    entries = [(0, i, e) for i, e in enumerate(cond.get("first_line", []))] + [(1, i, e) for i, e in enumerate(cond.get("second_line", []))]
    verdicts = screen_many(kb, [e["herb"] for _, _, e in entries], profile)
    usable = []
    for tier, idx, e in entries:
        v = verdicts[e["herb"]]
        if v.allowed:
            group = 1 if v.status == CAUTION else 0
            usable.append((group, tier, idx, PlanItem(e["herb"], e.get("prep", 0), e.get("why", ""), v, tier)))
        elif tier == 0 and len(plan.withheld) < 4:
            plan.withheld.append((_short_name(kb.herbs[e["herb"]]), v.avoid_reasons[:2]))
    usable.sort(key=lambda t: t[:3])
    plan.items = [u[3] for u in usable[:max_items]]
    return plan


def _card(kb: HerbalistKB, item: PlanItem, cond: dict[str, Any]) -> Recommendation:
    herb = kb.herbs[item.herb_id]
    prep = herb["prep"][min(item.prep_idx, len(herb["prep"]) - 1)]
    v = item.verdict
    precautions = list(dict.fromkeys(v.cautions + v.notes + herb.get("cautions", [])))
    avoid_if = []
    min_age = herb.get("min_age")
    if min_age:
        avoid_if.append(f"Age: not recommended below about {min_age} years" + (f". {herb['child_note']}" if herb.get("child_note") else ""))
    for key, label in (("preg", "Pregnancy"), ("bf", "Breastfeeding")):
        level = herb.get(key, "limited")
        extra = herb.get("preg_note" if key == "preg" else "bf_note") or ""
        avoid_if.append(f"{label}: {'avoid' if level == 'avoid' else 'ok' if level == 'ok' else 'only small amounts and with advice'}. {extra}".strip())
    for tag in herb.get("avoid_conditions", []):
        avoid_if.append(f"Not suitable with {tag}")
    for inter in herb.get("interactions", []):
        if inter.get("sev") in ("major", "moderate"):
            avoid_if.append(f"Interacts with {inter['cls'].replace('_', ' ')} medicines ({inter['sev']}): {inter.get('note', '')}")
    route = prep.get("route", "oral")
    return Recommendation(
        kind="herb", id=f"{item.herb_id}:{route}", name=f"{_short_name(herb)} ({prep.get('form', route)})",
        category=cond.get("category", "general"), why=item.why or herb.get("evidence_note", ""),
        evidence_level=herb.get("evidence", "unknown"), evidence_text=evidence_label(herb.get("evidence")), route=route,
        how_to_use=f"{prep.get('how', '')} Amount: {prep.get('amount', '')}, {prep.get('freq', '')}. For how long: {prep.get('duration', '')}.",
        precautions=precautions[:6], checks=list(dict.fromkeys(v.checks)), avoid_if=avoid_if,
        stop_when=list(herb.get("stop_if", [])), seek_care="; ".join(cond.get("see_doctor_if", [])[:3]),
        references=list(herb.get("sources", [])), herb_id=item.herb_id,
    )


def pick_questions(kb: HerbalistKB, profile: UserProfile, plan: Plan | None, cond: dict[str, Any] | None,
                   condition_id: str | None) -> list[ha.Question]:
    """Up to two short questions that would sharpen the advice. Only list-type question at a time so a bare 'no' is clear."""
    needed: set[str] = set()
    if plan:
        for it in plan.items:
            needed.update(it.verdict.needs)
    needed.add("age")  # amounts and suitability depend on age, so a herbalist always asks
    qs: list[ha.Question] = []
    list_added = 0
    for qid in ("age", "pregnancy", "medications", "conditions", "allergies"):
        if qid not in needed or ha._safety_satisfied(profile, qid) or profile.ask_counts.get(qid, 0) >= 2:
            continue
        q = ha.SAFETY_QUESTIONS[qid]
        if q.list_type:
            if list_added:
                continue
            list_added += 1
        qs.append(q)
        if len(qs) >= 2:
            break
    if len(qs) < 2 and cond and condition_id:
        for i, text in enumerate(cond.get("questions", [])):
            qid = f"c_{condition_id}_{i}"
            if qid not in profile.asked_questions:
                qs.append(ha.Question(qid, text, "concern"))
                break
    return qs


def _bullets(lines: list[str]) -> str:
    return "\n".join(f"- {x}" for x in lines)


def compose_plan_message(kb: HerbalistKB, profile: UserProfile, plan: Plan, triage: TriageInfo,
                         questions: list[ha.Question], updated: bool) -> str:
    cond = kb.condition(plan.condition_id) or {}
    out: list[str] = []
    if updated:
        out.append("Thanks, that helps. Here is your plan updated for what you've told me.")
    else:
        out.append(f"**{cond.get('title', plan.title)}.** {cond.get('overview', '')}")
    if triage.level == SEE_CLINICIAN:
        why = "; ".join(triage.labels) if triage.labels else cond.get("title", "this").lower()
        out.append(f"**Please also have this checked by a doctor soon** ({why}). The herbs below are for comfort and support, not a substitute for being examined.")
    elif triage.advise_check:
        out.append("Because of " + "; ".join(triage.advise_check) + ", please get it checked by a doctor as well as trying the steps below.")

    if plan.items:
        out.append("**Here is what I'd suggest:**")
        for i, it in enumerate(plan.items, 1):
            herb = kb.herbs[it.herb_id]
            prep = herb["prep"][min(it.prep_idx, len(herb["prep"]) - 1)]
            line = f"{i}. **{_short_name(herb)}**, as {prep.get('form', 'a preparation')}: {it.why} {prep.get('how', '')} Use: {prep.get('freq', '')}; for {prep.get('duration', 'a short course')}."
            notes = (it.verdict.cautions + it.verdict.notes)[:2]
            if notes:
                line += "\n   " + " ".join(f"_{n}_" for n in notes)
            out.append(line)
        out.append("_Full preparation steps, safety checks and references are in the details below._")
    else:
        out.append("None of the usual herbs for this suit your situation as it stands, so let's lean on the care steps below, and I can look for a gentler option if you tell me a little more.")

    if plan.withheld:
        out.append("**Left out for you:** " + "; ".join(f"{n} ({r[0] if r else 'not a good fit'})" for n, r in plan.withheld))
    care = cond.get("self_care", [])[:4]
    if care:
        out.append("**Food and lifestyle:**\n" + _bullets(care))
    sd = cond.get("see_doctor_if", [])[:4]
    if sd:
        out.append("**See a doctor if:**\n" + _bullets(sd))
    if questions:
        out.append("**A couple of quick questions so I can fine-tune this:**\n" + fmt_numbered(questions))
    return "\n\n".join(out)


# ============================================================================ personalised follow-up (built-in herbalist)

def _insights_for(session: SessionState, cond: dict[str, Any] | None, texts: list[str], profile: UserProfile | None = None,
                  limit: int = 2) -> list[str]:
    """Condition-specific remarks triggered by the patient's own words; each is given only once per conversation."""
    if not cond:
        return []
    blob = " ".join(texts).lower().replace("\u2019", "'")
    out: list[str] = []
    for ins in cond.get("insights", []):
        key = f"{session.active_condition}:{ins['id']}"
        if key in session.shown_insights:
            continue
        if re.search(ins["pattern"], blob, flags=re.I):
            session.shown_insights.append(key)
            out.append(ins["say"])
            if profile is not None and ins.get("answers") is not None:
                profile.mark_asked(f"c_{session.active_condition}_{ins['answers']}")  # the patient already answered that question
            if len(out) >= limit:
                break
    return out


_MED_FILLER = re.compile(r"^(?:(?:i\s*(?:am|'m|m)|im|just|only|some|taking|take|on|my|a|the|and|also|using|use|i)\s+)+", re.I)


def clean_med_name(name: str) -> str:
    n = re.sub(r"\s+", " ", (name or "").strip(" .,;:-"))
    n = _MED_FILLER.sub("", n)
    n = re.sub(r"\s+\d{1,2}$", "", n).strip(" .,;:-")  # trailing list number from "1. ... 2."
    return n


def tidy_medications(profile: UserProfile) -> None:
    """Remove filler words and duplicates from the medicine list (keeps the patient's own spelling)."""
    seen, out = set(), []
    for m in profile.medications:
        c = clean_med_name(m)
        if c and c.lower() not in seen:
            seen.add(c.lower())
            out.append(c)
    profile.medications[:] = out


def _acknowledge(session: SessionState, kb: HerbalistKB, profile: UserProfile) -> list[str]:
    """One specific sentence for each new fact the patient has given, saying what it changes (or doesn't)."""
    done = session.ack_done
    lines: list[str] = []

    def once(key: str) -> bool:
        if key in done:
            return False
        done.append(key)
        return True

    if profile.medications_status == NONE and once("meds_none"):
        lines.append("No regular medicines, so I can use the full range of herbs for you.")
    elif profile.medications and profile.medications_status in (PROVIDED, "unsure") and any(
            ("med:" + m.lower()) not in done for m in profile.medications):
        fresh_meds = [m for m in profile.medications if ("med:" + m.lower()) not in done]
        done.extend("med:" + m.lower() for m in fresh_meds)
        pic = classify_medications(UserProfile(medications=fresh_meds, medications_status=profile.medications_status), kb.rules)
        for med in pic.no_interaction:
            lines.append(f"**{med}** does not interact with these herbs, so nothing needs to change there.")
        for cls, meds in pic.by_class.items():
            lines.append(f"**{', '.join(meds)}** ({cls.replace('_', ' ')} medicine) can interact with some herbs, so I've screened every herb against it and flagged any that need care.")
        if pic.unclassified:
            lines.append(f"I don't recognise **{', '.join(pic.unclassified)}**, so I've kept to gentle herbs in food and tea amounts. Tell me what it's for and I can be more precise.")
    if profile.pregnancy_status == "neither" and once("preg_neither"):
        lines.append("Not pregnant or breastfeeding, so the full list stays open.")
    elif profile.pregnancy_status in ("pregnant", "breastfeeding") and once("preg_yes"):
        lines.append(f"Since you're {profile.pregnancy_status}, I've limited this to the herbs that are acceptable then.")
    if profile.age is not None and once(f"age:{profile.age}"):
        if profile.age < 12:
            lines.append(f"At {profile.age}, only mild kitchen herbs in food or weak teas are suitable; a paediatric pharmacist should confirm amounts.")
        elif profile.age < 18:
            lines.append(f"At {profile.age}, I've kept to low-risk herbs at roughly half to adult amounts.")
        elif profile.age >= 65:
            lines.append(f"At {profile.age}, I've started with gentler amounts and checked for medicine interactions.")
        else:
            lines.append(f"Thank you, at {profile.age} the standard adult amounts apply.")
    if profile.allergies_status == NONE and once("allergy_none"):
        lines.append("No known allergies, good.")
    elif profile.allergies and once("allergy:" + "|".join(profile.allergies).lower()):
        lines.append(f"I've checked every herb against your allergy to {', '.join(profile.allergies)}.")
    if profile.conditions_status == NONE and once("cond_none"):
        lines.append("No other health conditions, which keeps things simple.")
    elif profile.conditions and once("cond:" + "|".join(profile.conditions).lower()):
        lines.append(f"With {', '.join(profile.conditions)}, I've graded each herb for you and chosen gentler options where needed.")
    return lines


def _plan_line(kb: HerbalistKB, it: PlanItem) -> str:
    herb = kb.herbs[it.herb_id]
    prep = herb["prep"][min(it.prep_idx, len(herb["prep"]) - 1)]
    return f"- **{_short_name(herb)}** ({prep.get('form', 'preparation')}): {prep.get('amount', '')}, {prep.get('freq', '')}."


def compose_followup_message(session: SessionState, kb: HerbalistKB, profile: UserProfile, plan: Plan, triage: TriageInfo,
                             questions: list[ha.Question], cond: dict[str, Any], remarks: list[str]) -> str:
    """A reply to the patient's latest answer: say what it changed, don't re-send an unchanged plan."""
    out: list[str] = []
    facts = _acknowledge(session, kb, profile)
    now = [it.herb_id for it in plan.items]
    same = bool(session.last_plan) and now == session.last_plan
    removed = [h for h in session.last_plan if h not in now]
    added = [h for h in now if h not in session.last_plan]

    if facts or remarks:
        out.append("**What I've taken from your reply**\n" + _bullets(facts + remarks))
    if triage.level == SEE_CLINICIAN:
        why = "; ".join(triage.labels) if triage.labels else cond.get("title", "this").lower()
        out.append(f"**Please also have this checked by a doctor soon** ({why}).")

    if not plan.items:
        out.append("None of the usual herbs suit your situation as it stands, so let's lean on the care steps and I can look for a gentler option if you tell me more.")
    elif same:
        out.append("**Your herbs stay the same:**\n" + "\n".join(_plan_line(kb, it) for it in plan.items))
    else:
        change = []
        unsuitable = [h for h in removed if h in kb.herbs and screen_herb(kb, h, profile).status == AVOID]
        spare = [h for h in removed if h in kb.herbs and h not in unsuitable]
        if unsuitable:
            change.append("I've taken out " + ", ".join(_short_name(kb.herbs[h]) for h in unsuitable) + " because it doesn't suit you")
        if spare and added:
            change.append("now that I know more, " + ", ".join(_short_name(kb.herbs[h]) for h in added)
                          + " is a better fit than " + ", ".join(_short_name(kb.herbs[h]) for h in spare) + " (which is still fine if you prefer it)")
        elif added:
            change.append("added " + ", ".join(_short_name(kb.herbs[h]) for h in added))
        out.append(("**Here is your plan, adjusted** (" + "; ".join(change) + "):" if change else "**Here is your plan, updated:**")
                   + "\n" + "\n".join(_plan_line(kb, it) for it in plan.items))
    if plan.withheld:
        out.append("**Left out for you:** " + "; ".join(f"{n} ({r[0] if r else 'not a good fit'})" for n, r in plan.withheld))

    # something new each turn: the next care tips, then the next-step herbs
    care = cond.get("self_care", [])
    start = session.selfcare_shown
    fresh = care[start:start + 3]
    if fresh:
        session.selfcare_shown = start + len(fresh)
        out.append("**More that helps:**\n" + _bullets(fresh))
    elif same and f"next:{session.active_condition}" not in session.shown_insights:
        in_plan = set(now)
        nxt = []
        for e in cond.get("second_line", []) + cond.get("first_line", []):
            hid = e["herb"]
            if hid in in_plan or hid in nxt:
                continue
            v = screen_herb(kb, hid, profile)
            if v.allowed:
                nxt.append(hid)
        if nxt:
            session.shown_insights.append(f"next:{session.active_condition}")
            out.append("**If this doesn't help enough after a couple of weeks,** the next thing I'd try is "
                       + " or ".join(f"**{_short_name(kb.herbs[h])}**" for h in nxt[:2]) + ". Tell me how it goes and I'll switch you over.")
    if questions:
        out.append("**To tailor this further:**\n" + fmt_numbered(questions))
    elif not (facts or remarks or fresh):
        out.append("I have what I need from you for now. Tell me how the plan is working, or ask me about any herb or anything else that is bothering you.")
    else:
        out.append("That's everything I needed. Start with the plan above, and tell me how you get on or ask about any herb in it.")
    return "\n\n".join(out)


def fmt_numbered(questions) -> str:
    return "\n".join(f"{i}. {q.text}" for i, q in enumerate(questions, 1))


def compose_referral_message(kb: HerbalistKB, condition_ids: list[str], triage: TriageInfo) -> str:
    cond = kb.condition(condition_ids[0]) if condition_ids else None
    why = "; ".join(triage.labels) or "what you've described"
    out = [f"From what you've described ({why}), I'd like a doctor to look at this **soon** rather than treat it at home, because it can have causes that need an examination or tests."]
    if cond and cond.get("see_doctor_if"):
        out.append("**Signs that make it more urgent:**\n" + _bullets(cond["see_doctor_if"][:4]))
    out.append("**Helpful to bring to the appointment:** when it started, whether it is changing, what you have tried, and every medicine and supplement you use.")
    if cond and cond.get("self_care"):
        out.append("**Meanwhile, for comfort only:**\n" + _bullets(cond["self_care"][:3]))
    out.append("I'm happy to keep helping with comfort measures and diet while you arrange that. Tell me how it is going or what else is bothering you.")
    return "\n\n".join(out)


def compose_herb_answer(kb: HerbalistKB, profile: UserProfile, herb_ids: list[str]) -> tuple[str, list[Recommendation], list[dict[str, Any]]]:
    """'Is X safe for me?' answered from the monograph and the patient's own profile."""
    lines: list[str] = []
    recs: list[Recommendation] = []
    withheld: list[dict[str, Any]] = []
    for hid in herb_ids[:2]:
        herb = kb.herbs[hid]
        v = screen_herb(kb, hid, profile)
        name = _short_name(herb)
        if v.status == AVOID:
            lines.append(f"**{name}:** I would not use this one for you. " + " ".join(v.avoid_reasons[:2]))
            withheld.append({"name": name, "reasons": v.avoid_reasons[:2]})
            alt = _alternative_for(kb, hid, profile)
            if alt:
                lines.append(f"A gentler alternative with a similar purpose is **{alt}**; tell me what you want it for and I'll tailor a plan.")
            continue
        prep = herb["prep"][0]
        lines.append(f"**{name}:** {herb.get('evidence_note', '')} A typical way to use it is {prep.get('form', '')}: {prep.get('how', '')} ({prep.get('freq', '')}; {prep.get('duration', '')}).")
        extras = (v.cautions + v.notes)[:3]
        if extras:
            lines.append("For you specifically: " + " ".join(extras))
        else:
            lines.append("I found nothing in what you've told me that rules it out.")
        item = PlanItem(hid, 0, herb.get("evidence_note", ""), v, 0)
        recs.append(_card(kb, item, {"category": "general", "see_doctor_if": []}))
    return "\n\n".join(lines), recs, withheld


def _alternative_for(kb: HerbalistKB, herb_id: str, profile: UserProfile) -> str | None:
    """A herb used for the same conditions that screens clean for this patient."""
    for cond in kb.conditions.values():
        listed = [e["herb"] for e in cond.get("first_line", []) + cond.get("second_line", [])]
        if herb_id in listed:
            for hid in listed:
                if hid != herb_id and screen_herb(kb, hid, profile).status in ("ok", "note"):
                    return _short_name(kb.herbs[hid])
    return None


def welcome_prompt(kb: HerbalistKB) -> str:
    topics = ", ".join(t.lower() for t in kb.topic_titles()[:12])
    return ("Tell me what's troubling you and I'll put together a plan, the way I would in clinic: what is probably going on, "
            "which herbs and kitchen remedies suit you, how to use them, and when to see a doctor. "
            "A few details help, such as how long it has lasted, your age, and any medicines you take, but you can start with just the problem.\n\n"
            f"I can help with things like: {topics}, and more.")


# ============================================================================ AI path

@dataclass
class Consult:
    message: str
    triage: str = SELF_CARE
    recommended: list[str] = field(default_factory=list)
    left_out: list[str] = field(default_factory=list)
    updates: dict[str, Any] = field(default_factory=dict)


def parse_consult(raw: str | None) -> Consult | None:
    if not raw or not isinstance(raw, str):
        return None
    data = extract_json(raw)
    if data is not None and isinstance(data.get("message"), str) and data["message"].strip():
        triage = data.get("triage") if data.get("triage") in (SELF_CARE, SEE_CLINICIAN, URGENT) else SELF_CARE
        rec = [str(x) for x in data.get("herbs_recommended", []) if isinstance(x, (str, int))] if isinstance(data.get("herbs_recommended", []), list) else []
        left = [str(x) for x in data.get("herbs_left_out", []) if isinstance(x, (str, int))] if isinstance(data.get("herbs_left_out", []), list) else []
        upd = data.get("profile_updates") if isinstance(data.get("profile_updates"), dict) else {}
        return Consult(data["message"], triage, rec, left, upd)
    if "{" not in raw[:5] and len(raw.strip()) > 40:  # model answered in plain prose
        return Consult(raw.strip())
    return None


def _clean_items(values: Any, limit: int = 12) -> list[str]:
    items: list[str] = []
    if not isinstance(values, list):
        return items
    for v in values:
        if isinstance(v, str):
            s = re.sub(r"\s+", " ", v.strip().lower())[:60]
            if 2 <= len(s) and s not in items:
                items.append(s)
    return items[:limit]


def apply_profile_updates(profile: UserProfile, upd: dict[str, Any], user_texts: list[str]) -> None:
    """Take facts the model says the patient STATED. Anything that would loosen a restriction needs a cue in the patient's own words."""
    if not upd:
        return
    joined = " ".join(user_texts).lower()
    last = (user_texts[-1] if user_texts else "").lower()
    negation = re.search(r"\b(no|none|nothing|not|nil|n/?a|nope|don'?t|do not|never)\b", last) is not None

    age = upd.get("age")
    if isinstance(age, int) and 0 < age < 121 and re.search(r"(?<!\d)" + str(age) + r"(?!\d)", joined):
        profile.age = age
    sex = upd.get("sex")
    if sex in ("female", "male", "other") and profile.sex is None:
        cues = {"female": r"\b(female|woman|girl|lady|she|wife|mother|mum|mom)\b", "male": r"\b(male|man|boy|gentleman|he|husband|father)\b", "other": r".*"}
        if re.search(cues[sex], joined):
            profile.sex = sex
    preg = upd.get("pregnancy_status")
    if preg == "pregnant" and re.search(r"pregnan|expecting|weeks along", joined):
        profile.pregnancy_status = PREG_PREGNANT
    elif preg == "breastfeeding" and re.search(r"breast ?feed|nursing|lactat", joined):
        profile.pregnancy_status = PREG_BREASTFEEDING
    elif preg == "neither" and negation and profile.pregnancy_status in ("unknown", "prefer_not"):
        profile.pregnancy_status = PREG_NEITHER
    for key, status_attr in (("allergies", "allergies_status"), ("medications", "medications_status"), ("conditions", "conditions_status")):
        items = _clean_items(upd.get(key))
        if items:
            current = getattr(profile, key)
            for it in items:
                if it not in current:
                    current.append(it)
            setattr(profile, status_attr, PROVIDED)
        elif upd.get(f"{key}_none") is True and negation and getattr(profile, status_attr) != PROVIDED:
            setattr(profile, status_attr, NONE)


def _herb_card_block(kb: HerbalistKB, herb_id: str, verdict: HerbVerdict, cond_entry: dict[str, Any] | None) -> dict[str, Any]:
    herb = kb.herbs[herb_id]
    prep_idx = (cond_entry or {}).get("prep", 0)
    prep_list = herb.get("prep", [])
    chosen = [prep_list[prep_idx]] if prep_idx < len(prep_list) else prep_list[:1]
    other = [p for i, p in enumerate(prep_list[:3]) if i != prep_idx][:1]
    return {
        "herb_id": herb_id, "name": herb.get("name"), "risk": herb.get("risk"), "evidence": herb.get("evidence_note"),
        "use_for_this_problem": (cond_entry or {}).get("why"), "preparations": chosen + other,
        "key_cautions": herb.get("cautions", [])[:2],
        "pregnancy": herb.get("preg"), "pregnancy_note": herb.get("preg_note"), "min_age": herb.get("min_age"),
        "interacts_with": [f"{i['cls']}:{i['sev']}" for i in herb.get("interactions", [])],
        "screening": verdict.for_model(),
    }


def build_context(session: SessionState, text: str, kb: HerbalistKB, condition_ids: list[str], triage: TriageInfo,
                  herb_ids: list[str], verdicts: dict[str, HerbVerdict], cond_entries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    profile = session.profile
    history = [{"role": m["role"], "content": m["content"][:700]} for m in session.messages[-HISTORY_TURNS:]]
    protocols = []
    for cid in condition_ids:
        c = kb.condition(cid) or {}
        protocols.append({
            "id": cid, "title": c.get("title"), "overview": c.get("overview"), "triage_default": c.get("triage"),
            "first_line_herbs": [e["herb"] for e in c.get("first_line", [])], "second_line_herbs": [e["herb"] for e in c.get("second_line", [])],
            "food_and_lifestyle": c.get("self_care"), "see_a_doctor_if": c.get("see_doctor_if"), "useful_questions": c.get("questions"),
        })
    return {
        "patient_message": text,
        "conversation": history,
        "patient_profile": profile.model_view(),
        "plan_already_given": {"herbs": session.last_plan, "times": session.plan_count,
                               "note": "If plan_already_given.times > 0 this is a follow-up: do NOT repeat the plan. Respond to the new information."},
        "still_unknown": summarize_profile_gaps(profile),
        "warning_signs_found": {"level": triage.level, "signs": triage.labels, "also_advise_check": triage.advise_check},
        "matched_protocols": protocols,
        "candidate_herbs": [_herb_card_block(kb, hid, verdicts[hid], cond_entries.get(hid)) for hid in herb_ids],
        "child_and_pregnancy_policy": kb.rules.get("child_policy", {}).get("notes", {}),
    }


# ---------------------------------------------------------------------------- guardrails

_BANNED_PHRASES = re.compile(
    r"(significant medical condition.{0,60}cleared by your clinician|interaction information (is )?insufficient|"
    r"i can(no|')?t (give|provide|offer|recommend) (you )?(any )?(a )?(herb|solution|advice|remed))", re.I)

_HERBALIST_NEGATABLE = {"claims a cure", "guarantees an outcome", "tells user to stop medication"}


def scan_herbalist_text(text: str, patient_text: str = "") -> list[str]:
    """Output guard for consultation text. Unlike the strict guard, dosing language is allowed (a herbalist gives amounts)."""
    problems: list[str] = []
    patient = patient_text.lower()
    for pat, why in sc.DANGEROUS_PATTERNS:
        if why == "states a dose":
            continue
        for m in re.finditer(pat, text or "", flags=re.IGNORECASE):
            before = (text or "")[max(0, m.start() - 40):m.start()]
            if why in _HERBALIST_NEGATABLE and sc._NEGATION_BEFORE.search(before):
                continue
            if why == "appears to diagnose":
                disease = re.sub(r".*\b(a |an )?", "", m.group(0).lower()).strip()
                if disease and disease[:5] in patient:
                    continue  # repeating what the patient told us is not a diagnosis
            problems.append(why)
            break
    if _BANNED_PHRASES.search(text or ""):
        problems.append("uses a blanket-refusal phrase instead of a tailored alternative")
    return problems


def validate_consult(c: Consult, kb: HerbalistKB, profile: UserProfile, triage: TriageInfo, patient_text: str) -> tuple[list[str], list[str]]:
    """Returns (problems, unscreened herb names). Any problem means the reply must not be shown as is."""
    problems: list[str] = []
    message = c.message
    if triage.level != URGENT and len(message.strip()) < 40:
        problems.append("reply is too short to be useful")
    problems += scan_herbalist_text(message, patient_text)

    # banned herbs: in the list, or recommended in the text
    for hit in kb.never_recommend(" ".join(c.recommended)):
        problems.append(f"recommends {hit['name']} ({hit['why']})")
    for entry in kb.rules.get("never_recommend", []):
        for n in [entry["name"].split(" (")[0], *entry.get("aliases", [])]:
            if len(n) >= 4 and re.search(r"\b(use|take|try|drink|add|apply|start|swallow|consider)\b[^.\n]{0,30}\b" + re.escape(n.lower()), message.lower()):
                problems.append(f"recommends {entry['name']} ({entry['why']})")
                break

    # per-patient screening of recommended herbs
    picture = classify_medications(profile, kb.rules)
    tags = patient_condition_tags(profile, kb.rules)
    unscreened: list[str] = []
    seen: set[str] = set()
    for name in c.recommended:
        hid = kb.find_herb(name)
        if hid is None:
            if name not in unscreened:
                unscreened.append(name)
            continue
        if hid in seen:
            continue
        seen.add(hid)
        v = screen_herb(kb, hid, profile, picture, tags)
        if v.status == AVOID:
            problems.append(f"recommends {kb.herbs[hid]['name'].split(' (')[0]} but screening says avoid: {'; '.join(v.avoid_reasons[:2])}")
    # herbs presented as numbered/bold recommendations that the model forgot to list
    left = {kb.find_herb(n) for n in c.left_out}
    for hid in kb.herbs_mentioned(re.sub(r"\*\*.*?left out.*", "", message, flags=re.I)):
        if hid in seen or hid in left:
            continue
        bold = re.findall(r"\*\*([^*]{2,80})\*\*", message)
        if any(kb.find_herb(b) == hid for b in bold):
            v = screen_herb(kb, hid, profile, picture, tags)
            if v.status == AVOID:
                problems.append(f"presents {kb.herbs[hid]['name'].split(' (')[0]} as a recommendation but screening says avoid")
            seen.add(hid)

    # triage consistency
    if triage.level == SEE_CLINICIAN and c.triage == SELF_CARE:
        problems.append("warning signs were found, so triage cannot be self_care: advise seeing a clinician soon")
    return problems, unscreened


# ---------------------------------------------------------------------------- main AI call

def _ai_consult(session: SessionState, text: str, kb: HerbalistKB, ai: AIClient, condition_ids: list[str], triage: TriageInfo,
                herb_ids: list[str], cond_entries: dict[str, dict[str, Any]]) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    profile = session.profile
    notices: list[dict[str, str]] = []
    user_texts = _recent_user_texts(session, 12)
    patient_text = " ".join(user_texts)
    feedback: list[str] | None = None
    for attempt in range(2):
        verdicts = screen_many(kb, herb_ids, profile)
        context = build_context(session, text, kb, condition_ids, triage, herb_ids, verdicts, cond_entries)
        if feedback:
            context["your_previous_reply_was_rejected_because"] = feedback
            context["instruction"] = "Write the reply again, fixing every problem listed. Keep the same JSON format."
        try:
            raw = ai.consult(context)
        except AIError as err:
            return None, [{"level": "warning", "text": f"{err.user_message} Using my built-in herbalist instead."}]
        except Exception:  # a provider problem must never crash the chat
            return None, [{"level": "warning", "text": "The AI herbalist could not be reached. Using my built-in herbalist instead."}]
        c = parse_consult(raw)
        if c is None:
            feedback = ["The reply was not the required JSON object with a 'message' field."]
            continue
        apply_profile_updates(profile, c.updates, user_texts)
        problems, unscreened = validate_consult(c, kb, profile, triage, patient_text)
        if problems:
            feedback = problems
            continue
        content = sanitize_text(c.message)[:MAX_MESSAGE_CHARS]
        if unscreened and (profile.medications or profile.medications_status not in (NONE, "unanswered")):
            content += ("\n\n_I haven't run " + ", ".join(unscreened[:3]) + " through my interaction screen, so please have a pharmacist check "
                        "it against your medicines before you start._")
        recs, withheld = [], []
        shown: set[str] = set()
        picture, tags = classify_medications(profile, kb.rules), patient_condition_tags(profile, kb.rules)
        for name in c.recommended:
            hid = kb.find_herb(name)
            if hid and hid not in shown:
                shown.add(hid)
                v = screen_herb(kb, hid, profile, picture, tags)
                entry = cond_entries.get(hid) or {"prep": 0, "why": ""}
                cond = next((kb.condition(cid) for cid in condition_ids if kb.condition(cid)), {"category": "general", "see_doctor_if": []})
                recs.append(_card(kb, PlanItem(hid, entry.get("prep", 0), entry.get("why", ""), v, 0), cond))
        for name in c.left_out:
            hid = kb.find_herb(name)
            if hid:
                v = screen_herb(kb, hid, profile, picture, tags)
                if v.status == AVOID:
                    withheld.append({"name": _short_name(kb.herbs[hid]), "reasons": v.avoid_reasons[:2]})
        return _msg(content, mode="ai", recommendations=recs, withheld=withheld, triage=c.triage if c.triage != SELF_CARE else triage.level), notices
    notices.append({"level": "info", "text": "The AI herbalist's draft did not pass my safety checks, so I'm using my built-in herbalist for this reply."})
    return None, notices


# ============================================================================ entry point

def respond(session: SessionState, text: str, awaiting: list[str], kb: HerbalistKB, ai: AIClient | None) -> dict[str, Any]:
    """One herbalist turn (the emergency scan and profile update have already happened)."""
    profile = session.profile
    ai_on = bool(ai is not None and ai.configured and session.ai_consent)
    user_texts = _recent_user_texts(session)
    recent_text = " ".join(user_texts)

    # ---- greetings and thanks
    if _THANKS.match(text) and session.active_condition:
        return _msg("You're welcome. Start the plan today and give it the time mentioned, and come back to tell me how it goes, "
                    "or if anything changes or something new comes up.")
    if _GREETING.match(text) and len(text.split()) <= 4:
        session.awaiting = []
        return _msg("Hello, and welcome. I'm your herbalist. " + welcome_prompt(kb))

    # ---- what is the concern?
    herb_mentions = kb.herbs_mentioned(text)
    matches = kb.match_conditions(text)
    new_topic = bool(matches)
    if not matches:
        matches = kb.match_conditions(" ".join(user_texts[-3:]))
    condition_ids = [m.condition_id for m in matches]
    if not new_topic and session.active_condition:
        condition_ids = [session.active_condition]
    elif condition_ids:
        if new_topic:
            session.active_condition = condition_ids[0]
            session.plan_count = 0
    primary = condition_ids[0] if condition_ids else None

    # ---- herb suitability question ("is ginger safe for me?")
    if herb_mentions and _SUITABILITY.search(text) and not (new_topic and not any(h in text.lower() for h in ("safe", "suitable"))):
        if not ai_on:
            body, recs, withheld = compose_herb_answer(kb, profile, herb_mentions)
            screened = [screen_herb(kb, h, profile) for h in herb_mentions[:2]]
            need = {n for v in screened for n in v.needs}
            qs = [ha.SAFETY_QUESTIONS[q] for q in ("age", "pregnancy", "medications", "conditions") if q in need and not ha._safety_satisfied(profile, q)][:2]
            if qs:
                body += "\n\n**To check this properly for you:**\n" + fmt_numbered(qs)
                session.awaiting = ha.register_asked(profile, qs)
            return _msg(body, recommendations=recs, withheld=withheld)

    # ---- triage
    triage = assess_triage(session, kb, condition_ids, recent_text)
    if triage.level == URGENT:
        session.emergency_active = True
        label = triage.urgent_label or "a possible emergency"
        return _msg(sc.EMERGENCY_MESSAGE + f"\n\nWhat I noticed: {label}.", notices=[{"level": "emergency", "text": "Possible emergency detected. No herbal advice given."}], triage=URGENT)

    cond = kb.condition(primary) if primary else None
    notices: list[dict[str, str]] = []
    if triage.level == SEE_CLINICIAN:
        notices.append({"level": "warning", "text": "A doctor's assessment is recommended for this. Herbs here are for comfort and support only."})

    # ---- nothing recognised
    if not condition_ids and not herb_mentions:
        if ai_on:
            reply, ai_notices = _ai_consult(session, text, kb, ai, [], triage, [], {})
            if reply:
                reply["notices"] = notices + ai_notices
                return reply
            notices += ai_notices
        return _msg("I'd like to understand a little more. Can you tell me what you're noticing, where, how long it has been going on, and whether anything makes it better or worse? "
                    + welcome_prompt(kb), notices=notices)

    # ---- candidate herbs for the model's context
    cond_entries: dict[str, dict[str, Any]] = {}
    herb_ids: list[str] = []
    for cid in condition_ids[:2]:
        c = kb.condition(cid) or {}
        for e in c.get("first_line", []) + c.get("second_line", []):
            cond_entries.setdefault(e["herb"], e)
            if e["herb"] not in herb_ids:
                herb_ids.append(e["herb"])
    for h in herb_mentions:
        if h not in herb_ids:
            herb_ids.append(h)
    herb_ids = herb_ids[:14]

    # ---- AI herbalist
    if ai_on:
        reply, ai_notices = _ai_consult(session, text, kb, ai, condition_ids, triage, herb_ids, cond_entries)
        notices += ai_notices
        if reply:
            reply["notices"] = notices
            session.plan_count += 1
            return reply

    # ---- built-in herbalist
    if triage.level == SEE_CLINICIAN and any(l for l in triage.labels if l not in (cond or {}).get("title", "").lower()) and _hard_referral(triage, cond):
        return _msg(compose_referral_message(kb, condition_ids, triage), notices=notices, triage=SEE_CLINICIAN)

    tidy_medications(profile)
    for qid in awaiting:
        if str(qid).startswith("c_"):
            session.concern_answers[qid] = text[:300]
    plan = build_plan(kb, profile, primary)
    updated = session.plan_count > 0 and not new_topic
    if not updated:
        session.last_plan, session.selfcare_shown = [], 0
    remarks = _insights_for(session, cond, user_texts[-1:] if updated else user_texts[-1:], profile)
    questions = pick_questions(kb, profile, plan, cond, primary)
    session.awaiting = ha.register_asked(profile, questions)
    if updated and cond:
        content = compose_followup_message(session, kb, profile, plan, triage, questions, cond, remarks)
    else:
        content = compose_plan_message(kb, profile, plan, triage, questions, False)
        if cond:
            session.selfcare_shown = min(4, len(cond.get("self_care", [])))
            if remarks:
                content = content.replace("**Here is what I'd suggest:**", "**What you've told me points to this:**\n" + _bullets(remarks) + "\n\n**Here is what I'd suggest:**", 1)
    session.last_plan = [it.herb_id for it in plan.items]
    recs = [_card(kb, it, cond or {}) for it in plan.items]
    session.plan_count += 1
    return _msg(content, recommendations=recs, withheld=[{"name": n, "reasons": r} for n, r in plan.withheld],
                notices=notices, triage=triage.level)


def _hard_referral(triage: TriageInfo, cond: dict[str, Any] | None) -> bool:
    """Warning signs from the patient's words (not the mere nature of the condition) mean no herbal plan, just referral."""
    if not triage.labels:
        return False
    nature = (cond or {}).get("title", "").lower() + " usually needs a proper check"
    return any(l != nature for l in triage.labels)
