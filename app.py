"""Herbal Wellness AI - Streamlit UI. Business logic lives in modules/; this file only renders and routes input."""
from __future__ import annotations

import streamlit as st

import config
from modules import conversation_manager as cm
from modules.ai_client import AIClient
from modules.herbal_knowledge_base import KnowledgeBase
from modules.herbalist_kb import HerbalistKB
from modules.product_recommender import ProductCatalogue
from modules.session_manager import (
    NONE, PREFER_NOT, PREG_BREASTFEEDING, PREG_NA, PREG_NEITHER, PREG_PREFER_NOT, PREG_PREGNANT,
    PREG_UNKNOWN, PROVIDED, UNANSWERED, UNSURE, get_session, reset_session,
)
from utils import formatting as fmt
from utils.validators import parse_list_field, validate_age

st.set_page_config(page_title=config.APP_NAME, page_icon="🌿", layout="wide", initial_sidebar_state="expanded")

CSS = """
<style>
  .block-container {padding-top: 1.6rem; max-width: 980px;}
  h1, h2, h3 {color: #1F3D2B; letter-spacing: -0.01em;}
  [data-testid="stSidebar"] {background: #E4EBDC; border-right: 1px solid #CBD7C0;}
  [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {color: #1F3D2B;}
  .hw-wordmark {font-family: Georgia, 'Times New Roman', serif; font-size: 1.55rem; color: #1F3D2B; font-weight: 600; margin-bottom: 0;}
  .hw-tag {color: #5F7A63; font-size: 0.82rem; margin-top: -0.2rem; margin-bottom: 0.8rem;}
  .hw-hero {background: linear-gradient(135deg, #2F5D3A 0%, #4F7A55 100%); color: #F8F4EA; padding: 1.1rem 1.4rem; border-radius: 14px; margin-bottom: 1rem;}
  .hw-hero h2 {color: #F8F4EA; margin: 0 0 .2rem 0; font-family: Georgia, serif;}
  .hw-hero p {margin: 0; opacity: .92; font-size: .95rem;}
  div[data-testid="stVerticalBlockBorderWrapper"] {border-radius: 12px;}
  .stButton > button {border-radius: 999px; border: 1px solid #9DB39C; background: #F8F4EA; color: #1F3D2B;}
  .stButton > button:hover {border-color: #2F5D3A; color: #2F5D3A; background: #EEF3E8;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)  # static CSS only: no user or model text is ever injected as HTML


@st.cache_resource
def load_data() -> tuple[KnowledgeBase, ProductCatalogue, HerbalistKB]:
    return KnowledgeBase.load(), ProductCatalogue.load(), HerbalistKB.load()


kb, catalogue, hkb = load_data()
HERBALIST = config.herbalist_mode()
session = get_session(st.session_state)
ai = AIClient()
if HERBALIST and ai.configured and not st.session_state.get("consent_initialised"):
    # Herbalist replies are written by the AI, so it is on by default; the sidebar toggle still turns it off.
    session.ai_consent = True
    st.session_state["consent_initialised"] = True

# Handle new input FIRST so the sidebar profile panel and the chat below both show the updated state.
prompt = st.chat_input("Describe your concern, e.g. \"my hair has been falling out for three months\"", max_chars=1500)
queued = st.session_state.pop("queued_input", None)
user_text = prompt or queued
if user_text:
    with st.spinner("Thinking carefully..."):
        cm.handle_user_message(session, user_text, kb, catalogue, ai if session.ai_consent else None, hkb=hkb)

ROUTE_LABEL = {"topical": "Topical", "oral": "Oral", "inhaled": "Inhaled / aromatherapy", "food": "Food", "gargle": "Gargle"}
EVIDENCE_BADGE = {"well_established": "green", "moderate": "green", "limited": "orange", "preliminary": "orange", "traditional_only": "gray"}


# ------------------------------------------------------------------ rendering helpers

def render_recommendation(rec, key_prefix: str, nested: bool = False) -> None:
    """`nested` = already inside an expander (Streamlit forbids expanders inside expanders), so details show inline."""
    with st.container(border=True):
        st.markdown(f"#### {rec.name}")
        badges = [f":green-badge[{fmt.category_label(rec.category)}]",
                  f":{EVIDENCE_BADGE.get(rec.evidence_level, 'gray')}-badge[Evidence: {rec.evidence_text}]",
                  f":blue-badge[{ROUTE_LABEL.get(rec.route, 'Oral')}]"]
        if rec.is_demo:
            badges.append(":red-badge[Demonstration entry, not a real product]")
        st.markdown(" ".join(badges))
        st.write(rec.why)
        if rec.ingredients:
            st.caption("Ingredients (as listed in the catalogue): " + ", ".join(rec.ingredients))
        st.markdown("**Key precautions**")
        for p in rec.precautions[:4]:
            st.markdown(f"- {p}")
        details = st.container() if nested else st.expander("View details")
        with details:
            if nested:
                st.markdown("---")
            if rec.how_to_use:
                st.markdown(f"**How to use:** {rec.how_to_use}")
            if rec.avoid_if:
                st.markdown("**Who should avoid it or ask a professional first**")
                for a in rec.avoid_if:
                    st.markdown(f"- {a}")
            if rec.stop_when:
                st.markdown("**Stop using it if**")
                for s in rec.stop_when:
                    st.markdown(f"- {s}")
            st.markdown(f"**When to seek medical care:** {rec.seek_care}")
            if rec.precautions[4:]:
                st.markdown("**More precautions**")
                for p in rec.precautions[4:]:
                    st.markdown(f"- {p}")
            if rec.checks:
                st.markdown("**Checks performed for you**")
                for c in rec.checks:
                    st.markdown(f"- {c}")
            if rec.references:
                st.markdown("**References**")
                for r in rec.references:
                    st.markdown(f"- {r}")


def render_message(msg: dict, idx: int) -> None:
    with st.chat_message(msg["role"], avatar="🌿" if msg["role"] == "assistant" else None):
        for n in msg.get("notices", []):
            {"emergency": st.error, "warning": st.warning}.get(n["level"], st.info)(n["text"])
        st.markdown(msg["content"])
        if msg["role"] != "assistant":
            return
        if msg.get("recommendations"):
            if msg.get("cards_collapsed"):
                with st.expander(msg.get("plan_heading") or "Details for the herbs above"):
                    for i, rec in enumerate(msg["recommendations"]):
                        render_recommendation(rec, f"{idx}-{i}", nested=True)
            else:
                st.markdown("##### Options that passed the safety checks")
                for i, rec in enumerate(msg["recommendations"]):
                    render_recommendation(rec, f"{idx}-{i}")
        if msg.get("products"):
            st.markdown("##### Products from the catalogue")
            for i, rec in enumerate(msg["products"]):
                render_recommendation(rec, f"{idx}-p{i}")
        elif msg.get("no_product_message"):
            st.info(msg["no_product_message"])
        if msg.get("lifestyle"):
            with st.container(border=True):
                st.markdown("**Nutrition and lifestyle support**")
                for line in msg["lifestyle"]:
                    st.markdown(f"- {line}")
        if msg.get("seek_care") and (msg.get("recommendations") or msg.get("products")):
            st.warning("**When to seek medical care:** " + msg["seek_care"])
        if msg.get("withheld"):
            with st.expander("Options I held back, and why"):
                for w in msg["withheld"]:
                    st.markdown(f"**{w['name']}**")
                    for r in w["reasons"]:
                        st.markdown(f"- {r}")
        if msg.get("mode") == "ai":
            st.caption("Written by an AI herbalist and checked against the local herb knowledge base and safety rules. Not a substitute for a doctor or pharmacist." if HERBALIST
                       else "Wording written by an AI model and checked by safety validators. Recommendations come from the local knowledge base.")


# ------------------------------------------------------------------ sidebar

STATUS_OPTIONS = {"Not answered": UNANSWERED, "None": NONE, "Listed below": PROVIDED, "Not sure": UNSURE, "Prefer not to say": PREFER_NOT}
STATUS_LABELS = {v: k for k, v in STATUS_OPTIONS.items()}
PREG_OPTIONS = {"Not answered": PREG_UNKNOWN, "Not pregnant or breastfeeding": PREG_NEITHER, "Pregnant": PREG_PREGNANT,
                "Breastfeeding": PREG_BREASTFEEDING, "Not applicable": PREG_NA, "Prefer not to say": PREG_PREFER_NOT}
PREG_LABELS = {v: k for k, v in PREG_OPTIONS.items()}


def list_field_editor(label: str, status: str, items: list[str], key: str) -> tuple[str, list[str]]:
    st.markdown(f"**{label}**")
    s = st.selectbox(f"{label} status", list(STATUS_OPTIONS), index=list(STATUS_OPTIONS).index(STATUS_LABELS.get(status, "Not answered")),
                     key=f"{key}_status", label_visibility="collapsed")
    text = st.text_input(f"{label} list", value=", ".join(items), key=f"{key}_text", placeholder="separate items with commas",
                         label_visibility="collapsed")
    return STATUS_OPTIONS[s], parse_list_field(text)


with st.sidebar:
    st.markdown('<p class="hw-wordmark">🌿 Herbal Wellness AI</p><p class="hw-tag">Your AI herbalist</p>', unsafe_allow_html=True)
    if st.button("➕ New conversation", use_container_width=True):
        reset_session(st.session_state)
        st.rerun()

    st.markdown("##### Common concerns" if HERBALIST else "##### Wellness categories")
    cols = st.columns(2)
    topics = config.HERBALIST_TOPICS if HERBALIST else config.CATEGORIES
    for i, (key, label) in enumerate(topics.items()):
        if cols[i % 2].button(label, key=f"cat_{key}", use_container_width=True):
            st.session_state["queued_input"] = f"I'd like help with {label.lower()}."

    p = session.profile
    st.markdown("##### Your health profile")
    rows = fmt.profile_rows(p)
    if rows:
        for label, value in rows:
            st.markdown(f"**{label}:** {value}")
    else:
        st.caption("Nothing collected yet. I only keep what you tell me in this session.")

    with st.expander("Review or correct my details"):
        with st.form("profile_form"):
            age_text = st.text_input("Age", value="" if p.age is None else str(p.age))
            sex_opts = ["Not stated", "Female", "Male", "Other"]
            sex_cur = {"female": "Female", "male": "Male", "other": "Other"}.get(p.sex or "", "Not stated")
            sex = st.selectbox("Sex (only if relevant)", sex_opts, index=sex_opts.index(sex_cur))
            preg = st.selectbox("Pregnancy / breastfeeding", list(PREG_OPTIONS), index=list(PREG_OPTIONS).index(PREG_LABELS.get(p.pregnancy_status, "Not answered")))
            allergies_status, allergies = list_field_editor("Allergies / sensitivities", p.allergies_status, p.allergies, "pf_allergy")
            meds_status, meds = list_field_editor("Medicines & supplements", p.medications_status, p.medications, "pf_meds")
            cond_status, conds = list_field_editor("Medical conditions", p.conditions_status, p.conditions, "pf_cond")
            lifestyle = st.text_area("Daily life (diet, water, sleep, activity, stress, smoking/alcohol/caffeine)", value="" if p.lifestyle in (None, "not shared") else p.lifestyle, height=80)
            route_opts = ["No preference", "topical", "oral"]
            route = st.selectbox("Preferred route", route_opts, index=route_opts.index({"either": "No preference"}.get(p.preferences.get("route", ""), p.preferences.get("route", "No preference"))) if p.preferences.get("route", "No preference") in route_opts + ["either"] else 0)
            frag = st.checkbox("Fragrance-free products only", value=bool(p.preferences.get("fragrance_free")))
            saved = st.form_submit_button("Save changes", use_container_width=True)
        if saved:
            age, err = validate_age(age_text)
            if err:
                st.error(err)
            else:
                changed_safety = (p.age, p.allergies, p.medications, p.conditions, p.pregnancy_status) 
                p.age = age
                p.sex = None if sex == "Not stated" else sex.lower()
                p.lifestyle = lifestyle.strip()[:300] or p.lifestyle
                p.pregnancy_status = PREG_OPTIONS[preg]
                p.allergies_status, p.allergies = (allergies_status, allergies if allergies_status in (PROVIDED, UNSURE) else [])
                p.medications_status, p.medications = (meds_status, meds if meds_status in (PROVIDED, UNSURE) else [])
                p.conditions_status, p.conditions = (cond_status, conds if cond_status in (PROVIDED, UNSURE) else [])
                if allergies and allergies_status == UNANSWERED: p.allergies_status = PROVIDED
                if meds and meds_status == UNANSWERED: p.medications_status = PROVIDED
                if conds and cond_status == UNANSWERED: p.conditions_status = PROVIDED
                if route == "No preference":
                    p.preferences.pop("route", None)
                else:
                    p.preferences["route"] = route
                p.preferences["fragrance_free"] = frag
                if p.concern_category and session.stage >= 5:
                    with st.spinner("Re-checking safety with your updated details..."):
                        cm.reevaluate(session, kb, catalogue, ai if session.ai_consent else None, hkb=hkb)
                st.rerun()
    if st.button("🗑️ Clear my data now", use_container_width=True):
        reset_session(st.session_state)
        st.rerun()

    st.markdown("##### Settings")
    session.ai_consent = st.toggle(
        "Use AI to write replies", value=session.ai_consent, disabled=not ai.configured,
        help="When on, your recent messages and the profile details above are sent to the Anthropic API to word the replies. "
             "Safety checks and recommendations never depend on it.")
    if not ai.configured:
        st.caption("No API key found, so the app runs in guided mode.")

    with st.expander("About, privacy & disclaimer"):
        st.markdown(
            "**Not medical advice.** Herbal Wellness AI gives general educational information. It cannot diagnose, treat or replace a "
            "doctor or pharmacist, and herbs can cause allergies and interact with medicines.\n\n"
            "**Your data.** Your chat and health profile live only in this browser session's memory and disappear when you clear them "
            "or close the session. Nothing is written to a database or log by this app. If you switch on *Use AI*, your recent messages and "
            "relevant profile fields are sent to the Anthropic API; nothing is sent while it is off.\n\n"
            "**Limits.** The herb knowledge base is a compiled reference from general herbal literature and has not had a clinical review; "
            "herb amounts are typical kitchen and tea amounts and should be confirmed with a qualified herbalist or pharmacist, especially for children, "
            "pregnancy, or if you take medicines. Session privacy is not a "
            "compliance claim for any health-privacy law.")


# ------------------------------------------------------------------ main area

HERO_TEXT = ("Tell me what is troubling you. I will suggest herbs and kitchen remedies that suit you, how to use them, and when to see a doctor, "
             "checking them against your medicines and health conditions." if HERBALIST else
             "Tell me what you would like help with. I will ask a few questions, check safety first, "
             "and only then suggest options, honestly labelled by how much evidence supports them.")
st.markdown(f'<div class="hw-hero"><h2>Herbal Wellness AI</h2><p>{HERO_TEXT}</p></div>', unsafe_allow_html=True)

if not ai.configured:
    st.info(ai.status_message())
elif not session.ai_consent:
    st.info("AI-written replies are off. Turn on **Use AI to write replies** in the sidebar if you are comfortable sending your messages to the AI provider. Safety checks always run locally.")
if session.emergency_active:
    st.error("An emergency warning sign was mentioned earlier in this conversation. If you or someone else is in danger, contact your local emergency number now.")
if HERBALIST and (not hkb.loaded or hkb.problems):
    st.error("The herbalist knowledge base could not be fully loaded: " + "; ".join(hkb.problems[:3]))
if not kb.herbs:
    st.error("The herbal knowledge base could not be loaded (data/herbs.json). Recommendations are unavailable.")

with st.chat_message("assistant", avatar="🌿"):
    st.markdown(fmt.herbalist_welcome_message() if HERBALIST else fmt.welcome_message())

if not any(m["role"] == "user" for m in session.messages):
    st.markdown("**Ways to start**")
    for i, s in enumerate(fmt.HERBALIST_STARTERS if HERBALIST else fmt.STARTERS):
        if st.button(s, key=f"starter_{i}"):
            st.session_state["queued_input"] = s
            st.rerun()

for i, m in enumerate(session.messages):
    render_message(m, i)
