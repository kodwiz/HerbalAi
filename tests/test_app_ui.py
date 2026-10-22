"""End-to-end UI checks using Streamlit's headless AppTest (no browser, no network)."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


@pytest.fixture
def app(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("config.api_key", lambda: None)
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30)
    return at.run()


def send(at, text):
    at.chat_input[0].set_value(text).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_app_loads_and_shows_config_message_without_api_key(app):
    assert not app.exception
    assert any("guided mode" in i.value for i in app.info)
    assert any("Herbal Wellness AI" in m.value for m in app.markdown)


def test_hair_conversation_through_the_ui(app):
    for msg in ["My hair has been falling out for three months.", "female, 29", "spread evenly, no period problems; it runs in my family",
                "no itching or redness; nothing changed recently", "no allergies", "topical, not pregnant",
                "I eat mostly vegetarian, sleep 6h, stressed at work, walk daily, no smoking"]:
        send(app, msg)
    text = " ".join(m.value for m in app.markdown)
    assert "Options that passed the safety checks" in text
    assert "Key precautions" in text and "When to seek medical care" in text


def test_emergency_shows_error_banner(app):
    send(app, "I have chest pain and trouble breathing")
    assert any("emergency" in e.value.lower() for e in app.error)


def test_new_conversation_button_clears_session(app):
    send(app, "My hair has been falling out. I'm 30")
    assert app.session_state["hw_session"].profile.age == 30
    next(b for b in app.sidebar.button if "New conversation" in b.label).click().run()
    s = app.session_state["hw_session"]
    assert s.messages == [] and s.profile.age is None


def test_clear_my_data_button(app):
    send(app, "bloating. I'm 44, allergic to peanuts")
    next(b for b in app.sidebar.button if "Clear my data" in b.label).click().run()
    s = app.session_state["hw_session"]
    assert s.messages == [] and s.profile.allergies == []


def test_profile_panel_hides_unanswered_fields(app):
    side = " ".join(m.value for m in app.sidebar.markdown)
    assert "Allergies:" not in side and "Age group:" not in side
    send(app, "I'm 52 and my skin is dry")
    side = " ".join(m.value for m in app.sidebar.markdown)
    assert "Age group:" in side and "Allergies:" not in side


def test_starter_buttons_present_before_chat(app):
    assert len([b for b in app.button if b.label.startswith(("Help me", "I want", "What herbal", "Check whether"))]) >= 4
