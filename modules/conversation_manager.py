"""Orchestrates one conversational turn: safety first, then questions, then recommendations.

Stages: 1 understand concern, 2 identify missing info, 3 safety screening, 4 escalation / suitability,
5 recommendations, 6 follow-up.
"""
from __future__ import annotations

import re
from typing import Any

import config

from modules import health_assessment as ha
from modules import recommendation_engine as engine
from modules import safety_checker as sc
from modules.ai_client import AIClient, AIError
from modules.herbal_knowledge_base import KnowledgeBase, evidence_label
from modules.product_recommender import ProductCatalogue
from modules.response_validator import final_safety_pass, validate_model_response
from modules.session_manager import SessionState
from utils import formatting as fmt
from utils.validators import clean_user_text

HISTORY_TURNS = 12


def _assistant(content: str, **extra: Any) -> dict[str, Any]:
    msg = {"role": "assistant", "content": content, "recommendations": [], "products": [], "withheld": [],
           "notices": [], "lifestyle": [], "mode": "guided"}
    msg.update(extra)
    return msg


_NOT_PREGNANT = re.compile(r"\b(can'?t|cannot|can not|am not|i'?m not|not|no chance|impossible|unmarried|virgin)\b[^.]{0,20}\b(pregnant|pregnancy|expecting)\b", re.I)
_SYMPTOM_TALK = re.compile(r"\b(pain|painful|bleeding|heavy|worse|worsening|cramp|cramps|ache|itch|cough|fever|nausea|since|started|began|every (day|month|week))\b", re.I)
_PAST_AGE = re.compile(r"\b(?:was|were|at|since|from|when i was|age of|aged)\s+(?:the age of\s+)?(\d{1,2})\b", re.I)
_CURRENT_AGE = re.compile(r"\b(?:i am|i'm|im|age is|aged?|my age is)\s*:?\s*(\d{1,3})\b|\b(\d{1,3})\s*(?:years?|yrs?)[\s-]*old\b", re.I)


def _herbalist_fixups(profile, text: str, awaiting: list[str], prev_age) -> None:
    """Small corrections the generic intake parser gets wrong in a free-flowing consultation."""
    # "I was 13" (when it started) is not the patient's age
    if profile.age is not None and profile.age != prev_age:
        past = _PAST_AGE.search(text)
        if past and int(past.group(1)) == profile.age and not _CURRENT_AGE.search(text):
            profile.age = prev_age
    # an age statement is not a medical condition ("I'm 19" in reply to the conditions question)
    bare_age = re.compile(r"^\s*(?:i\s*am|i'?m|im)?\s*\d{1,3}(?:\s*(?:years?|yrs?)(?:\s*old)?)?\s*\.?\s*$", re.I)
    if profile.conditions and any(bare_age.match(c) for c in profile.conditions):
        profile.conditions[:] = [c for c in profile.conditions if not bare_age.match(c)]
        if not profile.conditions:
            profile.conditions_status = "unanswered"
    # a description of the symptom is not a list of medical conditions ("my pain is getting worse" in reply to the conditions question)
    if profile.conditions and "conditions" in awaiting and _SYMPTOM_TALK.search(text):
        from modules.herbalist import default_kb
        known = [kw for kws in default_kb().rules.get("condition_tags", {}).values() for kw in kws]
        if not any(sc._word_in(kw, text) for kw in known):
            profile.conditions[:] = []
            profile.conditions_status = "unanswered"
    # "I can't be pregnant" answers the pregnancy question
    if profile.pregnancy_status == "unknown" and _NOT_PREGNANT.search(text):
        profile.pregnancy_status = "neither"
        profile.mark_answered("pregnancy")


# ------------------------------------------------------------------ AI helper

def _try_ai(session: SessionState, ai: AIClient | None, kb: KnowledgeBase, task: str, extra: dict[str, Any],
            allowed_ids: set[str], user_mentioned: set[str]) -> tuple[str | None, list[dict[str, str]]]:
    """Ask the model for wording. Returns (validated_text | None, notices). Never raises."""
    if ai is None or not session.ai_consent or not ai.configured:
        return None, []
    history = [{"role": m["role"], "content": m["content"][:600]} for m in session.messages[-HISTORY_TURNS:]]
    context = {
        "profile": session.profile.model_view(),
        "conversation": history,
        "approved_herbs": kb.retrieval_context(sorted(allowed_ids)),
        "user_mentioned_herbs": sorted(user_mentioned),
        **extra,
    }
    try:
        raw = ai.complete_json(task, context)
    except AIError as err:
        return None, [{"level": "warning", "text": f"{err.user_message} Showing a guided reply instead."}]
    except Exception:  # never let a provider problem crash the app
        return None, [{"level": "warning", "text": "The AI reply could not be generated. Showing a guided reply instead."}]
    checked = validate_model_response(raw, allowed_ids, kb, user_mentioned)
    if not checked.ok:
        return None, [{"level": "info", "text": "An AI reply failed my safety validation and was not shown. Showing a guided reply instead."}]
    return checked.message, []


# ------------------------------------------------------------------ main entry

def handle_user_message(session: SessionState, text: str, kb: KnowledgeBase, catalogue: ProductCatalogue,
                        ai: AIClient | None = None, hkb: Any = None) -> dict[str, Any]:
    text = clean_user_text(text)
    profile = session.profile
    first_turn = not any(m["role"] == "user" for m in session.messages)
    session.messages.append({"role": "user", "content": text})
    if not text:
        reply = _assistant("I didn't catch anything there. Tell me what you'd like help with.")
        session.messages.append(reply)
        return reply

    # Stage 4 (always first): emergencies override everything
    hits = sc.scan_for_emergency(text)
    if hits:
        session.emergency_active = True
        session.stage = 4
        session.awaiting = []
        reply = _assistant(sc.emergency_response(hits), notices=[{"level": "emergency", "text": "Possible emergency detected. No herbal advice given."}])
        session.messages.append(reply)
        return reply

    awaiting = session.awaiting
    session.awaiting = []
    prev_age = profile.age
    updates = ha.apply_user_message(profile, text, awaiting)
    if config.herbalist_mode():
        _herbalist_fixups(profile, text, awaiting, prev_age)
    ha.settle_after_reply(profile, awaiting)

    if config.herbalist_mode():
        from modules import herbalist
        reply = herbalist.respond(session, text, awaiting, hkb or herbalist.default_kb(), ai)
        session.stage = 5
        session.messages.append(reply)
        return reply

    reply = (_handle_named_item(session, text, awaiting, kb, catalogue, ai)
             or _handle_followup(session, text, updates, kb, catalogue, ai)
             or _handle_main_flow(session, text, updates, first_turn, kb, catalogue, ai))
    session.messages.append(reply)
    return reply


# ------------------------------------------------------------------ named item ("is X suitable for me?")

_SUITABILITY = re.compile(r"\b(safe|suitable|okay|ok to|can i (use|take|have)|should i (use|take)|compatible|check whether)\b", re.I)


def _handle_named_item(session, text, awaiting, kb, catalogue, ai) -> dict | None:
    profile = session.profile
    starter = re.search(r"check whether a herbal product is suitable", text, re.I)
    if starter:
        session.awaiting = ["named_item"]
        return _assistant("Happy to check. Which herb or product is it? Please give the name, and the full ingredient list if you have it. "
                          "I can only assess herbs that are in my knowledge base, and I'll tell you plainly when I can't.")
    mentioned = kb.mentioned_in(text)
    asking = "named_item" in awaiting or session.pending_item is not None and not mentioned
    if not (mentioned and _SUITABILITY.search(text)) and not asking:
        return None
    if mentioned:
        name = mentioned[0]["common_name"]
        session.pending_item = None
    elif session.pending_item:
        name = session.pending_item
    else:
        name = text[:60]
    assessment = engine.assess_named_item(name, profile, kb)
    if not assessment.found:
        session.stage = 4
        session.pending_item = None
        return _assistant(assessment.message, notices=[{"level": "warning", "text": "Unknown ingredient: not assessed."}])
    session.stage = 4
    herb = assessment.herb
    missing: list[str] = []
    for _, result in assessment.verdicts:
        missing += result.missing_info
    # ask for essential info before judging
    if missing:
        routes = {r for r, res in assessment.verdicts if res.missing_info}
        qs = ha.next_questions(profile, routes or {"oral"}, max_questions=2)
        qs = [q for q in qs if q.kind == "safety"] or qs
        if qs:
            session.pending_item = herb["common_name"]
            session.awaiting = ha.register_asked(profile, qs)
            return _assistant(f"To check **{herb['common_name']}** safely for you I need a bit more information:\n\n{fmt.numbered(qs)}")
    session.pending_item = None
    lines = [f"**{herb['common_name']}**: evidence level is *{evidence_label(herb.get('evidence_level'))}*."]
    allowed_any = False
    for route, res in assessment.verdicts:
        way = "on skin/scalp" if route == "topical" else "by mouth"
        if res.allowed:
            allowed_any = True
            lines.append(f"- {way.capitalize()}: no conflict found with the information you gave me. This is not a guarantee of safety.")
            lines += [f"  - Caution: {c}" for c in res.cautions[:3]]
        else:
            lines.append(f"- {way.capitalize()}: **not suitable to recommend** at this time.")
            lines += [f"  - {b}" for b in res.blocks[:3]]
    if allowed_any:
        lines.append("Please confirm with a pharmacist or doctor if you take medicines or have a health condition.")
    return _assistant("\n".join(lines))


# ------------------------------------------------------------------ follow-up after recommendations

def _handle_followup(session, text, updates, kb, catalogue, ai) -> dict | None:
    result = session.last_result
    if session.stage != 6 or result is None:
        return None
    low = text.lower()
    meaningful = [u for u in updates if u.field not in ("trend", "preference")]
    if meaningful:
        return None  # new information: re-run the main flow so safety checks are repeated
    if any(f.action == "block_herbal" for f in sc.check_red_flags(session.profile)):
        return None  # new symptoms mentioned: escalate through the main flow
    cat = session.profile.concern_category or "general"
    if re.search(r"\b(doctor|physician|clinician|medical (care|help|advice)|when should i|seek)\b", low):
        return _assistant(f"**When to seek medical care:** {result.seek_care}\n\nStop any herb or product if you notice rash, swelling, "
                          "stomach pain, or anything unexpected, and seek urgent care for trouble breathing.")
    if re.search(r"\b(compare|difference|versus|vs\.?|which (one|is better))\b", low) and result.recommendations:
        lines = ["Here is how the options compare on evidence and key cautions:"]
        for r in result.recommendations:
            lines.append(f"- **{r.name}**: {r.evidence_text}. Main caution: {r.precautions[0] if r.precautions else 'see card'}")
        lines.append("Start with only one at a time so you can tell what helps or causes a reaction.")
        return _assistant("\n".join(lines))
    mentioned = {h["id"] for h in kb.mentioned_in(text)}
    allowed = result.allowed_herb_ids
    ai_text, notices = _try_ai(session, ai, kb, "answer_followup",
                               {"user_message": text, "approved_items": _approved_summary(result)}, allowed, mentioned & allowed)
    if ai_text:
        return _assistant(ai_text, mode="ai", notices=notices)
    return _assistant("I can explain any of the options above in more detail, compare them, or tell you when to see a doctor. "
                      "If anything about your health, medicines or allergies has changed, tell me and I'll re-check everything.",
                      notices=notices)


def _approved_summary(result) -> list[dict[str, Any]]:
    return [{"id": r.herb_id, "name": r.name, "route": r.route, "evidence": r.evidence_text, "why": r.why,
             "key_precautions": r.precautions[:2]} for r in result.recommendations]


# ------------------------------------------------------------------ stages 1-5

def _handle_main_flow(session, text, updates, first_turn, kb, catalogue, ai) -> dict:
    profile = session.profile

    if profile.concern_category is None:
        session.stage = 1
        pending = [ha.SAFETY_QUESTIONS[q] for q in ("sex", "age") if not ha._safety_satisfied(profile, q)]
        session.awaiting = ha.register_asked(profile, pending)
        return _assistant(fmt.no_start_message())

    flags = sc.check_red_flags(profile)
    if any(f.action == "block_herbal" for f in flags):
        session.stage = 4
        session.last_result = None
        return _assistant(fmt.escalation_message([f for f in flags if f.action == "block_herbal"], profile),
                          notices=[{"level": "warning", "text": "Medical evaluation recommended. Herbal suggestions are withheld."}])

    routes = engine.routes_needed(profile, kb)
    both_routes = {"oral", "topical"} <= {r for h in kb.candidates_for(profile.concern_category) for r in kb.routes_for(h, profile.concern_category)}
    if not ha.is_ready(profile, routes):
        questions = ha.next_questions(profile, routes, include_route_question=both_routes)
        if questions:
            session.stage = 3 if all(q.kind == "safety" for q in questions) else 2
            session.awaiting = ha.register_asked(profile, questions)
            ai_text, notices = _try_ai(session, ai, kb, "acknowledge_and_wait_for_answers",
                                       {"user_message": text, "questions_that_will_be_added_after_your_text": [q.text for q in questions]},
                                       set(), set())
            lead = ai_text or fmt.ack_and_questions_template(profile, [], first_turn).split("\n\n")[0]
            body = f"{lead}\n\n{fmt.numbered(questions)}\n\n_Answer in your own words or use the Health profile in the sidebar. \"Not sure\" and \"prefer not to say\" are fine._"
            return _assistant(body, mode="ai" if ai_text else "guided", notices=notices)

    return _recommend(session, text, kb, catalogue, ai)


def _recommend(session, text, kb, catalogue, ai) -> dict:
    profile = session.profile
    result = engine.run(profile, kb, catalogue)
    dropped = final_safety_pass(result, profile, kb, catalogue)
    session.last_result = result
    session.stage = 5
    notices: list[dict[str, str]] = []
    if result.clinician_advice:
        notices.append({"level": "warning", "text": "Your symptoms have lasted long enough that a clinician check is advisable alongside any self-care."})
    summary = _approved_summary(result)

    if not result.recommendations and not result.products:
        reasons = []
        for w in result.withheld[:4]:
            reasons.append(f"- **{w.name}**: {w.reasons[0]}")
        extra = ("\n\nWhat would help: " + ", ".join(_info_names(result.needs_info)) + ".") if result.needs_info else ""
        content = ("I'm not able to suggest a herbal option right now.\n\n" + ("\n".join(reasons) if reasons else
                   "I couldn't find an option in my knowledge base that is both relevant and safe for what you've told me.")
                   + extra + "\n\nA pharmacist or doctor is the best next step. Lifestyle ideas are below.")
        session.stage = 6
        return _assistant(content, lifestyle=result.lifestyle, notices=notices, withheld=_withheld_view(result), seek_care=result.seek_care)

    ai_text, ai_notices = _try_ai(session, ai, kb, "introduce_recommendations",
                                  {"approved_items": summary, "note": "Cards with full details are shown separately."},
                                  result.allowed_herb_ids, set())
    notices += ai_notices
    n = len(result.recommendations)
    intro = ai_text or (f"Based on what you've told me, {n} option{'s' if n != 1 else ''} passed my safety checks. "
                        "These are wellness options, not treatments or cures, and the strength of evidence varies. "
                        "Please start with only one, and read the precautions on each card.")
    session.stage = 6
    return _assistant(intro, mode="ai" if ai_text else "guided", recommendations=result.recommendations, products=result.products,
                      withheld=_withheld_view(result), lifestyle=result.lifestyle, notices=notices, seek_care=result.seek_care,
                      no_product_message=result.no_product_message, dropped=dropped.dropped)


def _withheld_view(result) -> list[dict[str, Any]]:
    return [{"name": w.name, "reasons": w.reasons[:2]} for w in result.withheld[:8]]


_INFO_LABELS = {"age": "your age", "allergies": "your allergies", "pregnancy": "pregnancy/breastfeeding status",
                "medications": "your medicines and supplements", "conditions": "your medical conditions"}


def _info_names(ids: list[str]) -> list[str]:
    return [_INFO_LABELS.get(i, i) for i in ids]


def reevaluate(session: SessionState, kb: KnowledgeBase, catalogue: ProductCatalogue, ai: AIClient | None = None,
               hkb: Any = None) -> dict[str, Any]:
    """Re-run all safety checks after the user edited their profile directly (no new chat text)."""
    if config.herbalist_mode():
        from modules import herbalist
        last_user = next((m["content"] for m in reversed(session.messages) if m["role"] == "user"), "")
        reply = herbalist.respond(session, last_user or "", [], hkb or herbalist.default_kb(), ai) if last_user else \
            {"role": "assistant", "content": "Thanks, I've noted your updated profile. Tell me what's bothering you and I'll tailor the plan.",
             "recommendations": [], "products": [], "withheld": [], "notices": [], "lifestyle": [], "mode": "guided"}
        reply["content"] = "I re-checked everything using your updated profile.\n\n" + reply["content"]
        session.messages.append(reply)
        return reply
    reply = _handle_main_flow(session, "", [], False, kb, catalogue, ai)
    reply["content"] = "I re-checked everything using your updated profile.\n\n" + reply["content"]
    session.messages.append(reply)
    return reply
