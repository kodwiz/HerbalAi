"""Structured, in-memory session state (health profile + conversation).

Nothing here is written to disk. Streamlit's session_state holds the object
for the lifetime of the browser session only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# Status for list-type safety fields.
UNANSWERED = "unanswered"
NONE = "none"            # user said they have none
PROVIDED = "provided"
UNSURE = "unsure"
PREFER_NOT = "prefer_not"
ANSWERED_STATUSES = {NONE, PROVIDED}

# Pregnancy status values
PREG_UNKNOWN = "unknown"
PREG_PREGNANT = "pregnant"
PREG_BREASTFEEDING = "breastfeeding"
PREG_NEITHER = "neither"          # not pregnant or breastfeeding
PREG_NA = "not_applicable"        # e.g. male / not relevant
PREG_PREFER_NOT = "prefer_not"


@dataclass
class UserProfile:
    concern_category: str | None = None
    concern_summary: str = ""
    age: int | None = None
    sex: str | None = None                 # "female" | "male" | "other" | None
    region: str | None = None
    pregnancy_status: str = PREG_UNKNOWN
    duration_days: int | None = None
    duration_text: str | None = None
    severity: str | None = None            # "mild" | "moderate" | "severe"
    trend: str | None = None               # "improving" | "worsening" | "unchanged"
    symptom_notes: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    conditions_status: str = UNANSWERED
    medications: list[str] = field(default_factory=list)
    medications_status: str = UNANSWERED
    allergies: list[str] = field(default_factory=list)
    allergies_status: str = UNANSWERED
    tried_remedies: list[str] = field(default_factory=list)
    lifestyle: str | None = None           # diet, water, sleep, activity, stress, smoking/alcohol/caffeine
    preferences: dict[str, Any] = field(default_factory=dict)  # route, fragrance_free, budget, diet, natural_vs_commercial
    asked_questions: list[str] = field(default_factory=list)   # question ids already asked
    answered_questions: list[str] = field(default_factory=list)
    ask_counts: dict[str, int] = field(default_factory=dict)

    # ---- helpers -----------------------------------------------------
    @property
    def age_group(self) -> str | None:
        if self.age is None:
            return None
        if self.age < 13:
            return "child"
        if self.age < 18:
            return "adolescent"
        if self.age < 65:
            return "adult"
        return "older adult"

    @property
    def is_minor(self) -> bool:
        return self.age is not None and self.age < 18

    @property
    def is_pregnant_or_breastfeeding(self) -> bool:
        return self.pregnancy_status in (PREG_PREGNANT, PREG_BREASTFEEDING)

    def all_symptom_text(self) -> str:
        return ". ".join(x for x in [self.concern_summary, *self.symptom_notes] if x).strip()

    def mark_asked(self, qid: str) -> None:
        if qid not in self.asked_questions:
            self.asked_questions.append(qid)

    def mark_answered(self, qid: str) -> None:
        if qid not in self.answered_questions:
            self.answered_questions.append(qid)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def model_view(self) -> dict[str, Any]:
        """Minimal profile sent to the AI model: only fields that matter, no empty values."""
        d = {
            "concern_category": self.concern_category,
            "concern_summary": self.concern_summary,
            "age_group": self.age_group,
            "sex": self.sex,
            "pregnancy_status": self.pregnancy_status if self.pregnancy_status != PREG_UNKNOWN else None,
            "duration": self.duration_text,
            "severity": self.severity,
            "trend": self.trend,
            "conditions": self.conditions or None,
            "medications": self.medications or None,
            "allergies": self.allergies or None,
            "tried_remedies": self.tried_remedies or None,
            "lifestyle": self.lifestyle,
            "preferences": self.preferences or None,
        }
        return {k: v for k, v in d.items() if v not in (None, "", [], {})}


@dataclass
class SessionState:
    profile: UserProfile = field(default_factory=UserProfile)
    messages: list[dict[str, Any]] = field(default_factory=list)  # {"role","content","recommendations"?,"safety"?}
    stage: int = 1
    awaiting: list[str] = field(default_factory=list)  # question ids asked in the last assistant turn
    emergency_active: bool = False
    ai_consent: bool = False
    pending_item: str | None = None      # herb/product the user asked about while safety info was missing
    last_result: Any = None              # most recent EngineResult (for follow-up questions)
    active_condition: str | None = None  # herbalist mode: protocol id currently being discussed
    plan_count: int = 0                  # herbalist mode: plans given for the active condition
    last_plan: list[str] = field(default_factory=list)       # herb ids in the last plan shown
    ack_done: list[str] = field(default_factory=list)        # facts already acknowledged to the patient
    shown_insights: list[str] = field(default_factory=list)  # personalised remarks already given
    selfcare_shown: int = 0                                  # how many care tips of the active condition were shown
    concern_answers: dict[str, str] = field(default_factory=dict)


SESSION_KEY = "hw_session"


def new_session() -> SessionState:
    return SessionState()


def get_session(store: dict | Any) -> SessionState:
    """Fetch (or create) the session from a dict-like store such as st.session_state."""
    if SESSION_KEY not in store:
        store[SESSION_KEY] = new_session()
    return store[SESSION_KEY]


def reset_session(store: dict | Any) -> SessionState:
    """Clear ALL health data and conversation for this browser session."""
    store[SESSION_KEY] = new_session()
    return store[SESSION_KEY]
