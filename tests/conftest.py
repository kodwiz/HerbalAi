import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from modules.herbal_knowledge_base import KnowledgeBase
from modules.session_manager import NONE, PROVIDED, PREG_NEITHER, UserProfile


@pytest.fixture(scope="session")
def kb():
    return KnowledgeBase.load()


@pytest.fixture
def adult():
    """A fully screened healthy adult: no allergies, meds, or conditions."""
    return UserProfile(
        concern_category="digestion", concern_summary="occasional bloating", age=34, sex="female",
        pregnancy_status=PREG_NEITHER, allergies_status=NONE, medications_status=NONE, conditions_status=NONE,
    )


LEGACY_TEST_MODULES = {"test_app_ui", "test_conversation_flow", "test_recommendation_engine", "test_response_validator", "test_safety_checker"}


@pytest.fixture(autouse=True)
def _strict_mode_for_legacy_tests(request, monkeypatch):
    """The original suite tests the strict intake flow; the herbalist suite sets its own mode."""
    if request.module.__name__.split(".")[-1] in LEGACY_TEST_MODULES:
        monkeypatch.setenv("HERBALIST_MODE", "0")
