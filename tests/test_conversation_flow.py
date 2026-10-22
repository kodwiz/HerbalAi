import json

import pytest

from modules import conversation_manager as cm
from modules import health_assessment as ha
from modules.ai_client import NO_KEY_MESSAGE, AIClient, AIError
from modules.product_recommender import ProductCatalogue
from modules.session_manager import SESSION_KEY, SessionState, UserProfile, get_session, reset_session


@pytest.fixture
def cat():
    return ProductCatalogue.load()


def chat(session, kb, cat, ai, *messages):
    reply = None
    for m in messages:
        reply = cm.handle_user_message(session, m, kb, cat, ai)
    return reply


class FakeProvider:
    def __init__(self, text=None, error=None):
        self.text, self.error, self.calls = text, error, []

    def complete(self, system, user_payload, max_tokens, timeout):
        self.calls.append(json.loads(user_payload))
        if self.error:
            raise self.error
        return self.text


# 8. missing API configuration
def test_missing_api_key_gives_clear_message():
    client = AIClient(api_key="")
    assert not client.configured
    assert "ANTHROPIC_API_KEY" in client.status_message() and "guided mode" in client.status_message()
    with pytest.raises(AIError) as err:
        client.complete_json("x", {})
    assert err.value.kind == "not_configured" and err.value.user_message == NO_KEY_MESSAGE


def test_placeholder_key_counts_as_unconfigured(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "your-api-key-here")
    import config
    assert config.api_key() is None


def test_app_works_without_ai_in_guided_mode(kb, cat):
    s = SessionState()
    reply = chat(s, kb, cat, AIClient(api_key=""), "My hair has been falling out for three months.")
    assert reply["mode"] == "guided" and "?" in reply["content"]


def test_ai_not_called_without_consent(kb, cat):
    provider = FakeProvider(text='{"message": "hi", "mentioned_herbs": []}')
    s = SessionState(ai_consent=False)
    chat(s, kb, cat, AIClient(provider=provider), "My hair has been falling out for three months.")
    assert provider.calls == []


def test_ai_receives_minimal_profile_and_no_empty_fields(kb, cat):
    provider = FakeProvider(text='{"message": "Thanks for telling me.", "mentioned_herbs": []}')
    s = SessionState(ai_consent=True)
    reply = chat(s, kb, cat, AIClient(provider=provider), "My hair has been falling out for three months.")
    assert reply["mode"] == "ai" and provider.calls
    sent = provider.calls[0]["profile"]
    assert all(v not in (None, "", [], {}) for v in sent.values())
    assert "allergies" not in sent  # not collected yet: not sent


# 9. malformed AI output never reaches the user
@pytest.mark.parametrize("bad", ["not json", '{"message": "Take ginger daily", "mentioned_herbs": ["ginger"]}', "", '{"message": 5}'])
def test_bad_model_output_falls_back_safely(kb, cat, bad):
    s = SessionState(ai_consent=True)
    reply = chat(s, kb, cat, AIClient(provider=FakeProvider(text=bad)), "My hair has been falling out for three months.")
    assert reply["mode"] == "guided" and "ginger" not in reply["content"].lower()
    assert any("guided reply" in n["text"] for n in reply["notices"])


@pytest.mark.parametrize("kind", ["timeout", "rate_limit", "network", "auth", "api_error"])
def test_api_failures_do_not_crash(kb, cat, kind):
    s = SessionState(ai_consent=True)
    err = AIError(kind, f"friendly {kind} message")
    reply = chat(s, kb, cat, AIClient(provider=FakeProvider(error=err)), "My hair has been falling out for three months.")
    assert reply["mode"] == "guided" and any(f"friendly {kind}" in n["text"] for n in reply["notices"])


def test_unexpected_provider_exception_does_not_crash(kb, cat):
    s = SessionState(ai_consent=True)
    reply = chat(s, kb, cat, AIClient(provider=FakeProvider(error=RuntimeError("boom"))), "bloating for a week")
    assert reply["content"]


# 2. emergencies end the herbal flow
def test_emergency_message_gets_escalation_not_herbs(kb, cat):
    s = SessionState()
    reply = chat(s, kb, cat, None, "I have chest pain and I can't breathe")
    assert "emergency" in reply["content"].lower() and not reply["recommendations"] and s.emergency_active


def test_emergency_beats_ai(kb, cat):
    provider = FakeProvider(text='{"message": "try ginger", "mentioned_herbs": []}')
    s = SessionState(ai_consent=True)
    chat(s, kb, cat, AIClient(provider=provider), "I think I'm having a stroke, my face is drooping")
    assert provider.calls == []


# 10 & 11. follow-up questions
def test_strict_intake_asks_sex_and_age_before_anything_else(kb, cat):
    s = SessionState()
    reply = chat(s, kb, cat, None, "My hair has been falling out for three months.")
    text = reply["content"].lower()
    assert "male or female" in text and "how old are you" in text
    assert not reply["recommendations"] and "allerg" not in text  # intake only, nothing else yet


def test_missing_info_triggers_relevant_followups(kb, cat):
    s = SessionState()
    chat(s, kb, cat, None, "My hair has been falling out for three months.")
    reply = chat(s, kb, cat, None, "I'm a 29 year old woman")
    text = reply["content"].lower()
    assert "hair" in text and "allerg" in text and "periods" in text      # sex-specific question for a woman
    assert "digest" not in text


def test_questions_differ_by_sex(kb, cat):
    f, m = SessionState(), SessionState()
    for sess, who in ((f, "woman"), (m, "man")):
        chat(sess, kb, cat, None, "My hair has been falling out for three months.")
        chat(sess, kb, cat, None, f"I'm a 29 year old {who}")
    assert "periods" in f.messages[-1]["content"].lower()
    assert "temples or crown" in m.messages[-1]["content"].lower() and "pregnant" not in m.messages[-1]["content"].lower()


def test_male_is_never_asked_about_pregnancy(kb, cat):
    s = SessionState()
    chat(s, kb, cat, None, "bloating for a week", "male", "I'm 35")
    for _ in range(4):
        chat(s, kb, cat, None, "no")
    assert "pregnant" not in " ".join(m["content"].lower() for m in s.messages if m["role"] == "assistant")
    assert s.profile.pregnancy_status == "not_applicable"


def test_male_pattern_baldness_is_not_read_as_sex(kb, cat):
    s = SessionState()
    chat(s, kb, cat, None, "I think I have male pattern baldness")
    assert s.profile.sex is None


def test_questions_are_specific_to_concern(kb, cat):
    s = SessionState()
    chat(s, kb, cat, None, "I get bloating and constipation")
    reply = chat(s, kb, cat, None, "I'm a 40 year old man")
    assert "blood" in reply["content"].lower() or "bloating" in reply["content"].lower()
    assert "scalp" not in reply["content"].lower()


def test_answered_questions_are_not_repeated(kb, cat):
    s = SessionState()
    chat(s, kb, cat, None, "My hair has been falling out for three months.")
    r2 = chat(s, kb, cat, None, "I'm a 31 year old woman, no allergies")
    text = r2["content"].lower()
    assert "male or female" not in text and "how old are you" not in text and "allergies or sensitivities" not in text
    r3 = chat(s, kb, cat, None, "spread evenly, no itching or redness")
    for phrase in ("male or female", "how old are you", "allergies or sensitivities", "spread evenly across"):
        assert phrase not in r3["content"].lower()


def test_no_question_is_asked_more_than_max_times(kb, cat):
    s = SessionState()
    chat(s, kb, cat, None, "bloating")
    for _ in range(8):
        chat(s, kb, cat, None, "hmm I am not sure what to say")
    assert max(s.profile.ask_counts.values()) <= ha.MAX_ASKS
    assert s.profile.allergies_status != "none"   # never silently assumed safe


def test_full_journey_ends_in_vetted_recommendations(kb, cat):
    s = SessionState()
    chat(s, kb, cat, None,
         "I have bloating after meals for a week",
         "male, 40",
         "mostly rice and lentils, with little water",
         "no allergies",
         "no medications",
         "no medical conditions",
         "I sleep 7 hours, walk daily, drink tea, no smoking or alcohol")
    last = [m for m in s.messages if m["role"] == "assistant" and m["recommendations"]]
    assert last, [m["content"][:80] for m in s.messages if m["role"] == "assistant"]
    assert all(r.route == "oral" for r in last[-1]["recommendations"])  # no skin products for digestion


def test_red_flag_in_followup_escalates(kb, cat):
    s = SessionState()
    chat(s, kb, cat, None, "My hair has been falling out for three months.", "female, 29", "spread evenly, no period problems; it runs in my family",
         "no itching or redness; nothing changed recently", "no allergies", "topical, not pregnant",
         "I eat mostly vegetarian, sleep 6h, stressed at work, walk daily, no smoking")
    assert s.stage == 6
    reply = chat(s, kb, cat, None, "Now I have bald patches and my scalp is bleeding")
    assert not reply["recommendations"] and "doctor" in reply["content"].lower()


def test_new_allergy_after_recommendations_rechecks(kb, cat):
    s = SessionState()
    chat(s, kb, cat, None, "My hair has been falling out for three months.", "female, 29", "spread evenly, no period problems; it runs in my family",
         "no itching or redness; nothing changed recently", "no allergies", "topical, not pregnant",
         "I eat mostly vegetarian, sleep 6h, stressed at work, walk daily, no smoking")
    reply = chat(s, kb, cat, None, "Actually I'm allergic to rosemary and mint")
    assert reply["recommendations"] or reply["withheld"]
    assert all(r.herb_id != "rosemary" for r in reply["recommendations"])


# named-item check
def test_named_unknown_ingredient_flow(kb, cat):
    s = SessionState()
    r1 = chat(s, kb, cat, None, "Check whether a herbal product is suitable for me.")
    assert "which herb or product" in r1["content"].lower() and s.awaiting == ["named_item"]
    r2 = chat(s, kb, cat, None, "Dragon's blood resin tonic")
    assert "can't tell you" in r2["content"]


def test_named_known_herb_is_checked_against_profile(kb, cat):
    s = SessionState()
    s.profile = UserProfile(age=35, sex="female", pregnancy_status="neither", allergies_status="none",
                            medications=["warfarin"], medications_status="provided", conditions_status="none")
    reply = chat(s, kb, cat, None, "Is ginger safe for me?")
    assert "not suitable" in reply["content"]


# 12. clearing the session
def test_reset_session_clears_everything(kb, cat):
    store = {}
    s = get_session(store)
    chat(s, kb, cat, None, "My hair has been falling out. I'm 30, allergic to peanuts")
    assert s.profile.age == 30 and s.messages
    fresh = reset_session(store)
    assert store[SESSION_KEY] is fresh
    assert fresh.messages == [] and fresh.profile.age is None and fresh.profile.allergies == [] and not fresh.emergency_active


# assessment parsing
def test_extraction_basics():
    p = UserProfile()
    ha.apply_user_message(p, "I'm 52, allergic to ragweed and lavender oil. I take metformin and vitamin D, diagnosed with type 2 diabetes")
    assert p.age == 52 and "ragweed" in p.allergies and "lavender oil" in p.allergies
    assert "metformin" in p.medications and "diabetes" in " ".join(p.conditions)


def test_bare_no_applies_only_to_first_open_question():
    p = UserProfile()
    ha.apply_user_message(p, "no", ["allergies", "medications"])
    assert p.allergies_status == "none" and p.medications_status == "unanswered"


def test_ambiguous_reply_is_not_recorded_as_allergy():
    p = UserProfile()
    ha.apply_user_message(p, "It's spread evenly, and I'm 29", ["hair_pattern", "age", "allergies"])
    assert p.allergies == [] and p.allergies_status == "unanswered"


def test_negated_pregnancy_not_recorded_as_pregnant():
    p = UserProfile()
    ha.apply_user_message(p, "I am not pregnant", ["pregnancy"])
    assert p.pregnancy_status == "neither"
