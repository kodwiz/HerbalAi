"""Adaptive questioning and gradual profile building.

* `classify_concern` decides which wellness area the user is talking about.
* `apply_user_message` pulls structured facts out of free text (deterministic, testable).
* `next_questions` picks only the questions that matter for THIS concern and
  never repeats a question that has been answered.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from modules.session_manager import (
    ANSWERED_STATUSES, NONE, PREFER_NOT, PREG_BREASTFEEDING, PREG_NA, PREG_NEITHER,
    PREG_PREFER_NOT, PREG_PREGNANT, PREG_UNKNOWN, PROVIDED, UNSURE, UserProfile,
)

# ------------------------------------------------------------------ concern classification

CONCERN_KEYWORDS = {
    "hair": ["hair", "scalp", "dandruff", "bald", "shedding", "alopecia"],
    "skin": ["skin", "acne", "pimple", "pigment", "dark spot", "eczema", "rash", "skincare", "wrinkle", "moistur", "complexion", "blemish"],
    "digestion": ["digest", "bloat", "constipat", "stomach", "gas", "indigestion", "diarrh", "bowel", "acidity", "nausea", "heartburn", "gut", "abdominal"],
    "sleep_stress": ["sleep", "insomnia", "stress", "anxious", "anxiety", "relax", "calm", "restless", "worry"],
    "nutrition": ["nutrition", "diet", "healthy eating", "protein", "vitamin", "meal", "weight", "calorie", "nutrient"],
    "immunity": ["immunity", "immune", "frequent cold", "catch cold", "flu", "infections"],
}


def classify_concern(text: str) -> str | None:
    t = (text or "").lower()
    scores = {cat: sum(1 for kw in kws if kw in t) for cat, kws in CONCERN_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    if scores[best] > 0:
        return best
    if re.search(r"\b(wellness|general health|healthy|health)\b", t):
        return "general"
    return None


# ------------------------------------------------------------------ question bank

@dataclass(frozen=True)
class Question:
    id: str
    text: str
    kind: str  # "concern" | "safety" | "preference"
    list_type: bool = False  # bare "no"/"none" is ambiguous when several are open


CONCERN_QUESTIONS: dict[str, list[Question]] = {
    "hair": [
        Question("hair_pattern", "Is the hair loss spread evenly across your scalp, or are there particular patches or a thinning area?", "concern"),
        Question("hair_scalp", "Do you have any scalp itching, redness, flaking, soreness or pain?", "concern"),
        Question("hair_triggers", "Did anything change around when it began, such as illness, major stress, a crash diet, childbirth or a new medication?", "concern"),
        Question("hair_history", "Does hair loss run in your family, and have you had hair loss before?", "concern"),
        Question("tried", "Have you tried any treatments or products so far, and did they help or cause any reaction?", "concern"),
    ],
    "skin": [
        Question("skin_type", "Which area of skin is affected, and what is it like (dry, oily, red, itchy, spots, dark patches)?", "concern"),
        Question("skin_alarm", "Is there any pain, oozing, swelling, spreading redness or a mole/spot that is changing?", "concern"),
        Question("skin_routine", "What does your current skincare routine include?", "concern"),
        Question("skin_history", "How long has this been going on, and do eczema, acne, allergies or asthma run in your family? How much sun exposure do you get?", "concern"),
        Question("tried", "Have you tried any treatments or products so far, and did they help or cause any reaction?", "concern"),
    ],
    "digestion": [
        Question("gi_symptoms", "What are you noticing (bloating, constipation, loose stools, gas, heartburn), and how often?", "concern"),
        Question("gi_alarm", "Have you had blood in stool or vomit, black stools, unexplained weight loss, trouble swallowing or severe pain?", "concern"),
        Question("gi_diet", "How would you describe your usual diet and fluid intake, including fibre-rich foods?", "concern"),
        Question("gi_timing", "Is it linked to meals, particular foods, stress, or time of day, and what is your usual bowel pattern?", "concern"),
        Question("tried", "Have you tried any remedies so far, and did they help or cause any reaction?", "concern"),
    ],
    "sleep_stress": [
        Question("sleep_pattern", "What does your sleep look like (trouble falling asleep, waking in the night, waking too early)?", "concern"),
        Question("sleep_caffeine", "How much caffeine or alcohol do you have, and at what times of day?", "concern"),
        Question("mood_check", "Have you been feeling persistently low or hopeless, or having panic attacks?", "concern"),
        Question("stress_source", "Is there a main source of stress or worry, and how much is it affecting your daily life and work?", "concern"),
        Question("tried", "Have you tried any remedies or routines so far, and did they help?", "concern"),
    ],
    "nutrition": [
        Question("nutrition_goal", "What is your main goal (more energy, digestion, a healthier eating pattern, a specific nutrient gap)?", "concern"),
        Question("nutrition_diet", "What does a typical day of eating look like, and do you follow any dietary pattern or restriction?", "concern"),
        Question("nutrition_screen", "How is your relationship with food right now, for example any restricting, bingeing or worry about eating?", "concern"),
        Question("nutrition_activity", "How active are you in a typical week, and has your weight or energy changed recently?", "concern"),
    ],
    "immunity": [
        Question("imm_symptoms", "What are you noticing (frequent colds, slow recovery, tiredness), and for how long?", "concern"),
        Question("imm_alarm", "Any fever, night sweats, swollen glands or unexplained weight loss?", "concern"),
        Question("imm_lifestyle", "How are your sleep, stress and diet at the moment?", "concern"),
        Question("imm_history", "Do you have any ongoing health conditions, or have you had recent illness, travel or hospital stays?", "concern"),
    ],
    "general": [
        Question("gen_goal", "What would you most like help with right now?", "concern"),
        Question("gen_lifestyle", "How are your sleep, diet and activity at the moment?", "concern"),
        Question("gen_symptoms", "Any specific symptoms you've noticed (energy, pain, mood, appetite), and how long have they been present?", "concern"),
        Question("gen_tried", "What have you already tried, and did it help?", "concern"),
    ],
}

SAFETY_QUESTIONS = {
    "sex": Question("sex", "To begin: are you male or female? (You can also say \"other\" or \"prefer not to say\".) Some herbs and risks differ by sex and hormonal status.", "safety"),
    "lifestyle": Question("lifestyle", "Tell me a little about daily life: diet, water intake, sleep, physical activity, stress, and whether you smoke or use alcohol or caffeine.", "safety"),
    "age": Question("age", "How old are you? (Age matters for safety.)", "safety"),
    "allergies": Question("allergies", "Do you have any allergies or sensitivities, especially to plants, herbs, essential oils, foods, fragrances or skin products? You can say \"none\", \"not sure\" or \"prefer not to say\".", "safety", True),
    "pregnancy": Question("pregnancy", "Are you pregnant or breastfeeding, or could you be? (Say \"no\" or \"not applicable\" if it does not apply.)", "safety", True),
    "medications": Question("medications", "Are you taking any prescription medicines, vitamins or supplements? Please list them, or say \"none\".", "safety", True),
    "conditions": Question("conditions", "Do you have any diagnosed medical conditions (for example liver, kidney, heart, thyroid, diabetes, autoimmune)? Say \"none\" if not.", "safety", True),
}
ROUTE_QUESTION = Question("route_pref", "Would you prefer options you apply on the skin/scalp, ones you take by mouth (tea, food, capsules), or either?", "preference")

# concern-specific questions that must be answered before recommending
MIN_CONCERN_ANSWERS = 4


# ------------------------------------------------------------------ extraction helpers

_NUM_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
              "eight": 8, "nine": 9, "ten": 10, "twelve": 12, "few": 3, "couple": 2, "several": 4}
_UNIT_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}
_NEG_START = re.compile(r"\b(no|not|none|nope|never|nothing|n/a|nil|without)\b", re.I)
_UNSURE = re.compile(r"\b(not sure|unsure|don'?t know|do not know|no idea|unknown)\b", re.I)
_PREFER_NOT = re.compile(r"\b(prefer not|rather not|don'?t want to (say|share|answer)|skip)\b", re.I)


def _split_items(text: str) -> list[str]:
    text = re.split(r"[.;!?\n]| but | and also |,?\s+and\s+(?:i|my)\s|,\s*(?:i|my)\s", text, maxsplit=1, flags=re.I)[0]
    parts = re.split(r",|/| and | & | plus ", text)
    items = []
    for p in parts:
        p = re.sub(r"^(?:some|any|a|an|the|my|also|mild|slight)\s+", "", p.strip().strip("\"'"), flags=re.I)
        p = re.sub(r"\b(allergy|allergies|allergic)\b", "", p, flags=re.I).strip(" -:")
        if 1 < len(p) <= 60 and not _NEG_START.fullmatch(p):
            items.append(p.lower())
    return items


def parse_duration_days(text: str) -> tuple[int | None, str | None]:
    t = text.lower()
    m = re.search(r"(?:for|over|about|around|nearly|almost|past|last)?\s*\b(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|twelve|few|couple|several)\b(?:\s+of)?\s*(day|week|month|year)s?\b", t)
    if m:
        n = m.group(1)
        n = int(n) if n.isdigit() else _NUM_WORDS.get(n, 1)
        return n * _UNIT_DAYS[m.group(2)], m.group(0).strip(" ,.")
    m = re.search(r"since\s+(last\s+)?(week|month|year|winter|summer|childhood)", t)
    if m:
        days = {"week": 7, "month": 30, "year": 365, "winter": 120, "summer": 120, "childhood": 3650}[m.group(2)]
        return days, m.group(0)
    return None, None


def parse_age(text: str, awaiting_age: bool) -> int | None:
    t = text.lower().strip()
    patterns = [
        r"\b(?:i am|i'm|im|age|aged|my age is)\s*(?:is\s*)?(\d{1,3})\b(?!\s*(?:kg|lb|cm|%|days?|weeks?|months?))",
        r"\b(\d{1,3})\s*[- ]?\s*(?:years?|yrs?|y/o|yo)[- ]?\s*old\b",
    ]
    if awaiting_age:
        patterns.append(r"\b(\d{1,3})\s*(?:years?|yrs?)\b")
        patterns.append(r"^\s*(\d{1,3})\s*[.!]?\s*$")
    for pat in patterns:
        m = re.search(pat, t)
        if m:
            age = int(m.group(1))
            if 0 < age < 121:
                return age
    if awaiting_age:  # e.g. "female, 29": a single standalone number is the answer to "how old are you?"
        nums = re.findall(r"(?<![\w.])(\d{1,3})(?![\w.%])", t)
        if len(nums) == 1 and 0 < int(nums[0]) < 121:
            return int(nums[0])
    return None


def _negated_near(text: str, word: str) -> bool:
    return bool(re.search(r"\b(not|no|never|isn'?t|am not|aren'?t|n't)\b[^.,;]{0,15}" + word, text, re.I))


@dataclass
class Update:
    field: str
    value: str


def apply_user_message(profile: UserProfile, text: str, awaiting: Iterable[str] = ()) -> list[Update]:
    """Update the profile from a user message. Returns what was noted (for transparency)."""
    awaiting = list(awaiting)
    updates: list[Update] = []
    raw = (text or "").strip()
    low = raw.lower()
    if not raw:
        return updates

    # concern
    if profile.concern_category is None:
        cat = classify_concern(raw)
        if cat:
            profile.concern_category = cat
            updates.append(Update("concern", cat))
    if not profile.concern_summary and len(raw) > 3 and classify_concern(raw):
        profile.concern_summary = raw[:300]
    profile.symptom_notes.append(raw[:500])
    profile.symptom_notes[:] = profile.symptom_notes[-12:]

    # age
    age = parse_age(raw, "age" in awaiting)
    if age is not None:
        profile.age = age
        updates.append(Update("age", str(age)))

    # sex: only from explicit statements, or when it was just asked ("male pattern baldness" must not count)
    m = (re.search(r"\b(?:i am|i'm|im)\s+(?:a\s+|an\s+)?(?:\d{1,3}[- ]?(?:year|yr)s?[- ]?old\s+)?(female|woman|girl|lady|male|man|boy|gentleman)\b", low)
         or re.search(r"\b\d{1,3}[- ]?(?:year|yr)s?[- ]?old\s+(female|woman|girl|male|man|boy)\b", low))
    if not m and "sex" in awaiting:
        m = re.search(r"\b(female|woman|girl|lady|male|man|boy|gentleman)\b", low)
        if not m and (_PREFER_NOT.search(low) or re.search(r"\b(other|non[- ]?binary|neither)\b", low)):
            profile.sex = "other"
            updates.append(Update("sex", "other"))
    if m:
        profile.sex = "female" if m.group(1) in ("female", "woman", "girl", "lady") else "male"
        updates.append(Update("sex", profile.sex))

    # lifestyle: free text, only when we asked, and only if it looks like an answer
    if "lifestyle" in awaiting and profile.lifestyle in (None, "not shared"):
        if re.search(r"\b(diet|eat|food|meal|sleep|exercise|walk|gym|active|sedentary|smok|alcohol|drink|coffee|tea|caffeine|water|stress|work|vegetarian|vegan)\b", low) and len(raw) > 12:
            profile.lifestyle = raw[:300]
            updates.append(Update("lifestyle", raw[:60]))
        elif _PREFER_NOT.search(low):
            profile.lifestyle = "prefer not to say"

    # duration
    days, dtxt = parse_duration_days(raw)
    if days is not None and (profile.duration_days is None or "duration" in awaiting):
        profile.duration_days, profile.duration_text = days, dtxt
        updates.append(Update("duration", dtxt or str(days)))

    # trend / severity
    if re.search(r"\b(worsening|getting worse|worse|deteriorat)", low):
        profile.trend = "worsening"; updates.append(Update("trend", "worsening"))
    elif re.search(r"\b(improving|getting better|better)\b", low) and not _negated_near(low, "better"):
        profile.trend = "improving"; updates.append(Update("trend", "improving"))
    elif re.search(r"\b(unchanged|same as before|stable|no change)\b", low):
        profile.trend = "unchanged"; updates.append(Update("trend", "unchanged"))
    m = re.search(r"\b(mild|moderate|severe)\b", low)
    if m:
        profile.severity = m.group(1)

    bare_neg = bool(re.fullmatch(r"\s*(no|nope|none|nothing|n/a|nil|nah|no thanks|none at all|not really)\s*[.!]?\s*", low))
    all_neg = bool(re.search(r"\b(no to (all|both|everything)|none of (these|the above)|neither)\b", low))
    open_lists = [q for q in awaiting if q in ("allergies", "pregnancy", "medications", "conditions")]
    first_open = open_lists[0] if open_lists else None
    # Treating a whole reply as a list of items is only safe when we asked nothing else.
    fallback_ok = first_open is not None and set(awaiting) <= set(SAFETY_QUESTIONS) and len(raw.split()) <= 14

    def _target(qid: str) -> bool:
        """Does a bare/ambiguous answer apply to this question?"""
        if qid in awaiting and (all_neg or (bare_neg and open_lists and open_lists[0] == qid)):
            return True
        return False

    # allergies
    if "allerg" in low or "sensitive to" in low or "reaction to" in low or first_open == "allergies":
        _apply_list_field(profile, raw, low, "allergies", (r"allerg(?:y|ies|ic)\s*(?:to|:|are|is)?", r"sensitive to", r"reaction to"), first_open == "allergies",
                          fallback=fallback_ok and first_open == "allergies",
                          target=_target("allergies"), updates=updates,
                          neg_patterns=[r"no (known )?(drug |food |plant |herbal )?allerg", r"nkda", r"not allergic to (anything|any)", r"(don'?t|do not) have (any )?allerg", r"allergy[- ]free"])

    # medications
    if re.search(r"\b(i take|i'm taking|i am taking|taking|i'm on|i am on|prescribed|medication|medicine|supplement|vitamin)", low) or first_open == "medications":
        _apply_list_field(profile, raw, low, "medications", (r"\bi take", r"i'm taking", r"i am taking", r"\btaking", r"i'm on\b", r"i am on\b", r"prescribed", r"medications?:", r"medicines?:"),
                          first_open == "medications", fallback=fallback_ok and first_open == "medications",
                          target=_target("medications"), updates=updates,
                          neg_patterns=[r"no (regular |current )?(medication|medicine|supplement|vitamin)s?", r"not (currently )?taking (any|anything)", r"(don'?t|do not) take (any|anything)", r"\bnothing\b"])

    # conditions
    if re.search(r"\b(diagnosed with|history of|suffer from|medical condition|i have (?:had )?(?:type|chronic|a history))", low) or first_open == "conditions":
        _apply_list_field(profile, raw, low, "conditions", (r"diagnosed with", r"history of", r"suffer from", r"medical conditions?:", r"\bi have"),
                          first_open == "conditions", fallback=fallback_ok and first_open == "conditions",
                          target=_target("conditions"), updates=updates,
                          neg_patterns=[r"no (known |medical |health |chronic )*(condition|issue|problem|illness)s?", r"(i'?m|i am) (generally )?(healthy|well)", r"nothing (medical|serious)"])

    # pregnancy / breastfeeding
    if profile.pregnancy_status in (PREG_UNKNOWN, PREG_PREFER_NOT) or first_open == "pregnancy":
        status = None
        if re.search(r"\bbreast ?feeding|\bnursing\b|\blactating\b", low) and not _negated_near(low, r"(breast ?feeding|nursing|lactating)"):
            status = PREG_BREASTFEEDING
        elif re.search(r"\bpregnan|\bexpecting\b", low) and not _negated_near(low, r"(pregnan|expecting)"):
            status = PREG_PREGNANT
        elif re.search(r"\bnot (currently )?(pregnant|breast ?feeding)|\bnot applicable\b|\bn/?a\b|\bdoes not apply\b|\bnot relevant\b", low) or _target("pregnancy"):
            status = PREG_NA if re.search(r"not applicable|n/?a\b|does not apply|not relevant|\bman\b|\bmale\b", low) else PREG_NEITHER
        elif first_open == "pregnancy" and _PREFER_NOT.search(low):
            status = PREG_PREFER_NOT
        elif profile.sex == "male":
            status = PREG_NA
        if status and status != profile.pregnancy_status:
            profile.pregnancy_status = status
            updates.append(Update("pregnancy", status))

    # route preference
    if "route_pref" in awaiting or re.search(r"\b(topical|apply|oral|tea|capsule|supplement|drink|eat)\b", low):
        pref = None
        if re.search(r"\b(either|both|no preference|any)\b", low):
            pref = "either"
        elif re.search(r"\b(topical|on the (skin|scalp)|apply|applied|cream|oil|serum|gel)\b", low) and not re.search(r"\b(oral|by mouth|tea|capsule)\b", low):
            pref = "topical"
        elif re.search(r"\b(oral|by mouth|tea|capsule|drink|eat|food|supplement)\b", low) and not re.search(r"\b(topical|on the (skin|scalp)|apply)\b", low):
            pref = "oral"
        if pref and (("route_pref" in awaiting) or "route" not in profile.preferences):
            profile.preferences["route"] = pref
            updates.append(Update("route preference", pref))

    # preferences
    if re.search(r"fragrance[- ]free|unscented|sensitive skin", low):
        profile.preferences["fragrance_free"] = True
        updates.append(Update("preference", "fragrance-free"))
    m = re.search(r"\b(vegan|vegetarian|halal|kosher|gluten[- ]free|dairy[- ]free)\b", low)
    if m:
        profile.preferences["diet"] = m.group(1)

    # tried remedies
    if "tried" in awaiting or re.search(r"\bi (have )?tried\b", low):
        if not bare_neg:
            profile.tried_remedies.append(raw[:200])
            updates.append(Update("tried", raw[:60]))

    return updates


def _apply_list_field(profile: UserProfile, raw: str, low: str, field_name: str, triggers: tuple[str, ...],
                      awaiting: bool, target: bool, updates: list[Update], neg_patterns: list[str],
                      fallback: bool = False) -> None:
    status_attr = f"{field_name}_status"
    items_attr = field_name
    if any(re.search(p, low) for p in neg_patterns) or target:
        if getattr(profile, status_attr) != PROVIDED:  # do not erase items the user already listed
            setattr(profile, status_attr, NONE)
            setattr(profile, items_attr, [])
            updates.append(Update(field_name, "none"))
        return
    if _PREFER_NOT.search(low):
        if awaiting:
            setattr(profile, status_attr, PREFER_NOT); updates.append(Update(field_name, "prefer not to say"))
        return
    unsure = bool(_UNSURE.search(low)) and awaiting
    if unsure:
        # keep anything the user did name, but the overall status stays "not sure" (never assumed safe)
        raw = _UNSURE.split(raw)[0] if _UNSURE.search(raw) else raw
        fallback = bool(raw.strip(" ,.-"))
    # find items after a trigger phrase; fall back to the whole reply if we asked directly
    segment = None
    for trig in triggers:
        m = re.search(trig, low)
        if m:
            segment = raw[m.end():]
            segment = re.sub(r"^\s*(to|of|with|from|by|:|are|is)\s+", "", segment, flags=re.I)
            break
    if segment is None and fallback:
        segment = raw
    if segment is None:
        if unsure:
            setattr(profile, status_attr, UNSURE)
            updates.append(Update(field_name, "not sure"))
        return
    items = _split_items(segment)
    if field_name == "medications":
        items = [re.sub(r"\s+(for|because|since|daily|twice|once)\b.*$", "", i) for i in items]
    if items:
        current = getattr(profile, items_attr)
        for it in items:
            if it not in current:
                current.append(it)
        setattr(profile, status_attr, UNSURE if unsure else PROVIDED)
        updates.append(Update(field_name, ", ".join(items)))
    elif unsure:
        setattr(profile, status_attr, UNSURE)
        updates.append(Update(field_name, "not sure"))


# ------------------------------------------------------------------ question selection

def _safety_satisfied(profile: UserProfile, qid: str) -> bool:
    if qid == "sex":
        return profile.sex is not None
    if qid == "lifestyle":
        return profile.lifestyle is not None
    if qid == "age":
        return profile.age is not None
    if qid == "allergies":
        return profile.allergies_status != "unanswered"
    if qid == "pregnancy":
        return profile.pregnancy_status != PREG_UNKNOWN
    if qid == "medications":
        return profile.medications_status != "unanswered"
    if qid == "conditions":
        return profile.conditions_status != "unanswered"
    return True


def concern_bank(profile: UserProfile) -> list[Question]:
    """Concern-specific questions, adjusted for sex where it changes what matters."""
    qs = list(CONCERN_QUESTIONS.get(profile.concern_category or "general", []))
    cat = profile.concern_category or "general"
    if profile.sex == "female" and cat in ("hair", "skin", "general"):
        qs.insert(min(1, len(qs)), Question("hormonal", "Any irregular or heavy periods, recent childbirth, menopause, PCOS or thyroid problems?", "concern"))
    if profile.sex == "male" and cat == "hair":
        qs.insert(min(1, len(qs)), Question("hair_male_pattern", "Is the thinning mainly at the temples or crown, and have you noticed it in other family members?", "concern"))
    return qs


def _concern_satisfied(profile: UserProfile, q: Question) -> bool:
    if q.id in profile.answered_questions:
        return True
    if q.id == "tried" and profile.tried_remedies:
        return True
    return False


def concern_progress(profile: UserProfile) -> tuple[int, int]:
    """(answered, needed) concern-specific questions."""
    qs = concern_bank(profile)
    answered = sum(1 for q in qs if _concern_satisfied(profile, q))
    if profile.duration_days is not None:
        answered += 1  # the user already gave onset/duration voluntarily
    # areas where warning signs are common get one extra screening question before any suggestion
    needed = min(MIN_CONCERN_ANSWERS, len(qs) + (1 if profile.duration_days is not None else 0))
    return answered, needed


def pregnancy_relevant(profile: UserProfile) -> bool:
    if profile.sex == "male":
        return False
    if profile.age is not None and (profile.age < 12 or profile.age > 55):
        return False
    return True


MAX_ASKS = 3  # a safety question still unanswered after this many tries is recorded as "not sure" (never assumed safe)


def next_questions(profile: UserProfile, routes_needed: set[str], include_route_question: bool = False,
                   max_questions: int = 4) -> list[Question]:
    """Questions to ask now (strict intake). Never repeats answered ones; caps repeat attempts."""
    # 0. Patient intake comes first: sex and age, before anything else
    intake = [SAFETY_QUESTIONS[q] for q in ("sex", "age")
              if not _safety_satisfied(profile, q) and profile.ask_counts.get(q, 0) < MAX_ASKS]
    if intake:
        return intake
    out: list[Question] = []
    # 1. concern-specific details (up to two per turn) until enough for a precise answer
    answered, needed = concern_progress(profile)
    if answered < needed:
        for q in concern_bank(profile):
            if not _concern_satisfied(profile, q) and q.id not in profile.asked_questions:
                out.append(q)
                if len(out) >= 2:
                    break
    # 2. preference
    if include_route_question and "route" not in profile.preferences and ROUTE_QUESTION.id not in profile.asked_questions and not out:
        out.append(ROUTE_QUESTION)
    # 3. safety essentials (only one list-type question per turn so a bare "no" is unambiguous)
    need_ids = ["allergies"]
    if pregnancy_relevant(profile):
        need_ids.append("pregnancy")
    if "oral" in routes_needed:
        need_ids += ["medications", "conditions"]
    list_added = 0
    for qid in need_ids:
        if _safety_satisfied(profile, qid):
            continue
        q = SAFETY_QUESTIONS[qid]
        if profile.ask_counts.get(qid, 0) >= MAX_ASKS:
            continue  # handled by settle_after_reply
        if q.list_type:
            if list_added >= 1:
                continue
            list_added += 1
        out.append(q)
        if len(out) >= max_questions:
            break
    # 4. daily-life question last and on its own, so it is not asked (and ignored) alongside others
    if not out and not _safety_satisfied(profile, "lifestyle") and profile.ask_counts.get("lifestyle", 0) < MAX_ASKS:
        out.append(SAFETY_QUESTIONS["lifestyle"])
    return out[:max_questions]


def register_asked(profile: UserProfile, questions: list[Question]) -> list[str]:
    ids = []
    for q in questions:
        profile.mark_asked(q.id)
        profile.ask_counts[q.id] = profile.ask_counts.get(q.id, 0) + 1
        ids.append(q.id)
    return ids


def settle_after_reply(profile: UserProfile, awaiting: list[str]) -> None:
    """After the user replied: concern questions count as answered; unresolved safety
    questions that were asked twice are recorded as 'unsure' (never assumed safe)."""
    for qid in awaiting:
        if qid in SAFETY_QUESTIONS:
            if _safety_satisfied(profile, qid):
                profile.mark_answered(qid)
            elif profile.ask_counts.get(qid, 0) >= MAX_ASKS and qid not in ("age", "sex"):
                if qid == "allergies": profile.allergies_status = UNSURE
                elif qid == "medications": profile.medications_status = UNSURE
                elif qid == "conditions": profile.conditions_status = UNSURE
                elif qid == "pregnancy": profile.pregnancy_status = PREG_PREFER_NOT
                elif qid == "lifestyle": profile.lifestyle = "not shared"
                elif qid == "sex": profile.sex = None
        else:
            profile.mark_answered(qid)


def safety_info_complete(profile: UserProfile, routes_needed: set[str]) -> bool:
    ids = ["sex", "age", "allergies"] + (["pregnancy"] if pregnancy_relevant(profile) else [])
    if "oral" in routes_needed:
        ids += ["medications", "conditions"]
    ids.append("lifestyle")
    return all(_safety_satisfied(profile, i) for i in ids)


def is_ready(profile: UserProfile, routes_needed: set[str], named_item_request: bool = False) -> bool:
    if not named_item_request:
        answered, needed = concern_progress(profile)
        if answered < needed:
            return False
    return safety_info_complete(profile, routes_needed)
