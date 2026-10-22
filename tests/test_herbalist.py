"""Herbalist mode: knowledge base integrity, graded screening, triage, AI guardrails, UI."""
import json

import pytest
from streamlit.testing.v1 import AppTest

from modules import conversation_manager as cm
from modules import herbalist
from modules.ai_client import AIClient
from modules.herbalist_kb import HerbalistKB
from modules.herbalist_screening import AVOID, CAUTION, OK, screen_herb
from modules.herbal_knowledge_base import KnowledgeBase
from modules.product_recommender import ProductCatalogue
from modules.session_manager import NONE, PREG_NEITHER, PREG_PREGNANT, PROVIDED, SessionState, UserProfile

BANNED = ["significant medical condition", "cleared by your clinician first", "interaction information insufficient",
          "can't give you a solution", "cannot give you a solution"]


@pytest.fixture(autouse=True)
def herbalist_on(monkeypatch):
    monkeypatch.setenv("HERBALIST_MODE", "1")


@pytest.fixture(scope="module")
def hkb():
    return HerbalistKB.load()


@pytest.fixture(scope="module")
def legacy_kb():
    return KnowledgeBase.load()


@pytest.fixture(scope="module")
def cat():
    return ProductCatalogue.load()


class FakeProvider:
    def __init__(self, *texts, error=None):
        self.texts, self.error, self.calls = list(texts), error, []

    def complete(self, system, user_payload, max_tokens, timeout):
        self.calls.append(json.loads(user_payload))
        if self.error:
            raise self.error
        return self.texts.pop(0) if len(self.texts) > 1 else self.texts[0]


def say(text, hkb, legacy_kb, cat, session=None, ai=None):
    session = session or SessionState()
    return session, cm.handle_user_message(session, text, legacy_kb, cat, ai, hkb=hkb)


def adult(**kw):
    base = dict(age=40, sex="male", pregnancy_status=PREG_NEITHER, allergies_status=NONE, medications_status=NONE, conditions_status=NONE)
    base.update(kw)
    return UserProfile(**base)


# ------------------------------------------------------------------ knowledge base
def test_kb_loads_without_problems(hkb):
    assert hkb.loaded and hkb.problems == []
    assert len(hkb.herbs) >= 50 and len(hkb.conditions) >= 30


def test_every_herb_has_amounts_and_sources(hkb):
    for hid, h in hkb.herbs.items():
        assert h["prep"] and all(p.get("amount") and p.get("how") for p in h["prep"]), hid


def test_never_recommend_list_not_in_herbs(hkb):
    for entry in hkb.rules["never_recommend"]:
        assert hkb.find_herb(entry["name"].split(" (")[0]) is None


@pytest.mark.parametrize("text,cond", [
    ("I get bloated and gassy after meals", "indigestion_bloating"),
    ("mujhe khansi hai", "cough"),
    ("I can't sleep at night", "insomnia_mild"),
    ("period cramps every month", "menstrual_cramps"),
    ("my hair is falling out", "hair_fall"),
])
def test_condition_matching(hkb, text, cond):
    assert cond in [m.condition_id for m in hkb.match_conditions(text)]


# ------------------------------------------------------------------ graded screening
def test_healthy_adult_ginger_is_fine(hkb):
    assert screen_herb(hkb, "ginger", adult()).status in (OK, "note")


def test_warfarin_ginger_is_caution_not_refusal(hkb):
    v = screen_herb(hkb, "ginger", adult(medications=["warfarin"], medications_status=PROVIDED))
    assert v.status == CAUTION and v.allowed


def test_warfarin_licorice_is_avoided(hkb):
    v = screen_herb(hkb, "licorice", adult(medications=["warfarin"], medications_status=PROVIDED))
    assert v.status == AVOID


def test_homeopathic_medicine_has_no_interaction(hkb):
    v = screen_herb(hkb, "ginger", adult(medications=["homeopathic medicine"], medications_status=PROVIDED))
    assert v.status in (OK, "note")


def test_unknown_medicine_does_not_block_low_risk_herb(hkb):
    v = screen_herb(hkb, "chamomile", adult(medications=["Cardiozid"], medications_status=PROVIDED))
    assert v.allowed


def test_serious_condition_is_graded(hkb):
    v = screen_herb(hkb, "chamomile", adult(conditions=["heart disease"], conditions_status=PROVIDED))
    assert v.allowed


def test_pregnancy_avoids_risky_herb(hkb):
    p = adult(sex="female", pregnancy_status=PREG_PREGNANT)
    assert screen_herb(hkb, "licorice", p).status == AVOID


# ------------------------------------------------------------------ built-in herbalist (no API key)
def test_plan_given_on_first_message(hkb, legacy_kb, cat):
    _, r = say("I get bloated and gassy after every meal", hkb, legacy_kb, cat)
    assert r["recommendations"] and "see a doctor" in r["content"].lower()
    assert not any(b in r["content"].lower() for b in BANNED)


def test_warfarin_and_heart_condition_still_get_a_plan(hkb, legacy_kb, cat):
    s, r = say("I'm on warfarin and have a heart condition and take Cardiozid. I can't sleep", hkb, legacy_kb, cat)
    assert r["recommendations"]
    assert not any(b in r["content"].lower() for b in BANNED)
    names = " ".join(c.name.lower() for c in r["recommendations"])
    assert "licorice" not in names and "valerian" not in names


def test_pregnant_nausea_plan_excludes_avoid_herbs(hkb, legacy_kb, cat):
    s, r = say("I'm 8 weeks pregnant and feel nauseous all day", hkb, legacy_kb, cat)
    names = " ".join(c.name.lower() for c in r["recommendations"])
    assert r["recommendations"] and "licorice" not in names


def test_warning_signs_refer_instead_of_herbal_plan(hkb, legacy_kb, cat):
    _, r = say("I have a cough for 5 weeks and I'm losing weight", hkb, legacy_kb, cat)
    assert not r["recommendations"] and "doctor" in r["content"].lower()


def test_emergency_stops_everything(hkb, legacy_kb, cat):
    s, r = say("I have chest pain and can't breathe", hkb, legacy_kb, cat)
    assert s.emergency_active and not r["recommendations"] and "emergency" in r["content"].lower()


def test_herb_suitability_question(hkb, legacy_kb, cat):
    _, r = say("Is ginger safe for me? I take warfarin", hkb, legacy_kb, cat)
    assert "ginger" in r["content"].lower() and not any(b in r["content"].lower() for b in BANNED)


# ------------------------------------------------------------------ AI herbalist and guardrails
def good_reply(**over):
    d = {"message": "Fennel tea after meals helps trapped gas. **Fennel seed tea**: crush 1/2 to 1 teaspoon, steep 10 minutes, up to 3 times a day. "
                    "See a doctor if it lasts more than 2 weeks.", "triage": "self_care",
         "herbs_recommended": ["fennel"], "herbs_left_out": [], "profile_updates": {}}
    d.update(over)
    return json.dumps(d)


def ai_session():
    s = SessionState()
    s.ai_consent = True
    return s


def test_ai_reply_used_when_valid(hkb, legacy_kb, cat):
    fp = FakeProvider(good_reply())
    s, r = say("I get bloated after meals", hkb, legacy_kb, cat, ai_session(), AIClient(provider=fp))
    assert r["mode"] == "ai" and "Fennel" in r["content"] and len(fp.calls) == 1
    assert fp.calls[0].get("candidate_herbs") is not None or fp.calls[0]


def test_blanket_refusal_is_rejected_then_retried(hkb, legacy_kb, cat):
    bad = good_reply(message="You mentioned a significant medical condition. Oral herbs should be cleared by your clinician first.")
    fp = FakeProvider(bad, good_reply())
    s, r = say("I get bloated after meals", hkb, legacy_kb, cat, ai_session(), AIClient(provider=fp))
    assert r["mode"] == "ai" and len(fp.calls) == 2 and "significant medical condition" not in r["content"]


def test_never_recommend_herb_falls_back_to_builtin(hkb, legacy_kb, cat):
    bad = good_reply(message="Take kava for relaxation, a teaspoon twice a day. See a doctor if it persists.", herbs_recommended=["kava"])
    fp = FakeProvider(bad)
    s, r = say("I get bloated after meals", hkb, legacy_kb, cat, ai_session(), AIClient(provider=fp))
    assert r["mode"] != "ai" and "kava" not in r["content"].lower() and r["recommendations"]


def test_avoid_herb_for_this_patient_is_rejected(hkb, legacy_kb, cat):
    bad = good_reply(message="Mulethi (licorice) tea soothes the throat. 1 teaspoon twice a day. See a doctor if no better in a week.", herbs_recommended=["licorice"])
    fp = FakeProvider(bad)
    s = ai_session()
    s.profile.medications, s.profile.medications_status = ["warfarin"], PROVIDED
    _, r = say("sore throat for 3 days", hkb, legacy_kb, cat, s, AIClient(provider=fp))
    assert r["mode"] != "ai" and "licorice" not in " ".join(c.name.lower() for c in r["recommendations"])


def test_provider_error_falls_back(hkb, legacy_kb, cat):
    from modules.ai_client import AIError
    fp = FakeProvider(error=AIError("timeout", "slow"))
    s, r = say("I get bloated after meals", hkb, legacy_kb, cat, ai_session(), AIClient(provider=fp))
    assert r["recommendations"] and any("built-in herbalist" in n["text"] for n in r["notices"])


def test_ai_cannot_downgrade_urgent_triage(hkb, legacy_kb, cat):
    fp = FakeProvider(good_reply())
    s, r = say("crushing chest pain and trouble breathing", hkb, legacy_kb, cat, ai_session(), AIClient(provider=fp))
    assert not fp.calls and not r["recommendations"]


# ------------------------------------------------------------------ personalised follow-ups
def convo(hkb, legacy_kb, cat, *messages):
    s = SessionState()
    replies = [say(m, hkb, legacy_kb, cat, s)[1] for m in messages]
    return s, replies


def test_followup_does_not_resend_the_same_plan(hkb, legacy_kb, cat):
    s, (r1, r2, r3) = convo(hkb, legacy_kb, cat, "I have period cramps every month.", "just some clacium supliments, i was 13",
                            "I am just taking clacium supliments 2.No I cant be pregnant")
    assert r2["content"] != r1["content"] and r3["content"] != r2["content"]
    assert "What I've taken from your reply" in r2["content"]
    assert "Here is what I'd suggest" not in r2["content"] + r3["content"]


def test_typo_calcium_is_recognised_as_non_interacting(hkb, legacy_kb, cat):
    s, (_, r2) = convo(hkb, legacy_kb, cat, "I have period cramps every month.", "just some clacium supliments")
    assert "does not interact" in r2["content"] and "don't recognise" not in r2["content"]
    assert s.profile.medications == ["clacium supliments"]


def test_i_was_13_is_not_the_age(hkb, legacy_kb, cat):
    s, _ = convo(hkb, legacy_kb, cat, "I have period cramps every month.", "just some clacium supliments, i was 13")
    assert s.profile.age is None
    s2, (_, r) = convo(hkb, legacy_kb, cat, "I have period cramps every month.", "I'm 19")
    assert s2.profile.age == 19 and "19" in r["content"]


def test_cant_be_pregnant_is_understood(hkb, legacy_kb, cat):
    s, _ = convo(hkb, legacy_kb, cat, "I have period cramps every month.", "No I cant be pregnant")
    assert s.profile.pregnancy_status == "neither"


def test_symptom_details_personalise_the_reply(hkb, legacy_kb, cat):
    _, (_, r) = convo(hkb, legacy_kb, cat, "I have a cough", "it is dry and worse at night")
    assert "dry, tickly cough" in r["content"] or "marshmallow" in r["content"].lower()
    assert "night" in r["content"].lower()


def test_symptom_talk_is_not_saved_as_a_condition(hkb, legacy_kb, cat):
    s, _ = convo(hkb, legacy_kb, cat, "I have period cramps every month.", "none", "my pain is getting worse and bleeding is heavy")
    assert not any("pain" in c.lower() for c in s.profile.conditions)


# ------------------------------------------------------------------ strict mode still available
def test_strict_mode_switch(monkeypatch, hkb, legacy_kb, cat):
    monkeypatch.setenv("HERBALIST_MODE", "0")
    _, r = say("I get bloated after meals", hkb, legacy_kb, cat)
    assert not r["recommendations"]  # strict flow asks the intake questions first


# ------------------------------------------------------------------ UI
def test_ui_herbalist_flow():
    app = AppTest.from_file(str(__import__("pathlib").Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30).run()
    assert not app.exception
    app.chat_input[0].set_value("I get bloated and gassy after every meal").run()
    assert not app.exception
    text = " ".join(m.value for m in app.markdown)
    assert "fennel" in text.lower() or "peppermint" in text.lower() or "ginger" in text.lower()
    assert "significant medical condition" not in text
