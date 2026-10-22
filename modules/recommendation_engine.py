"""Turns (profile + knowledge base + catalogue + safety rules) into vetted recommendations.

The AI model never decides WHAT is recommended; it can only help explain what this
engine has already approved. Safety results always win over relevance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from modules import safety_checker as sc
from modules.herbal_knowledge_base import KnowledgeBase, evidence_label
from modules.product_recommender import ProductCatalogue, ProductExclusion, ProductMatch, match_products
from modules.session_manager import UserProfile

EVIDENCE_RANK = {"well_established": 0, "moderate": 1, "limited": 2, "preliminary": 3, "traditional_only": 4}
MAX_HERB_RECOMMENDATIONS = 3

LIFESTYLE = {
    "digestion": ["Eat regular meals and add fibre-rich foods (oats, fruit, vegetables, legumes, whole grains) gradually.",
                  "Spread fluids through the day and keep a short food-and-symptom diary to spot triggers.",
                  "Gentle daily movement, such as a walk, often helps digestion."],
    "hair": ["Avoid tight hairstyles and very hot styling; use a mild shampoo.",
             "Eat enough protein and varied, iron-containing foods. Crash diets can themselves trigger shedding.",
             "If shedding persists, ask a clinician whether tests (for example iron or thyroid) are worthwhile."],
    "skin": ["Use a gentle cleanser, a fragrance-free moisturiser and daily sunscreen.",
             "Introduce only one new product at a time and patch test it first.",
             "Avoid picking or scrubbing, which can worsen irritation and marks."],
    "sleep_stress": ["Keep consistent sleep and wake times and limit caffeine after midday.",
                     "Build a screen-free wind-down routine and get daylight and movement during the day.",
                     "Slow breathing or other relaxation practice can help some people."],
    "nutrition": ["Build meals around vegetables, fruit, whole grains, legumes and a protein source, and drink water regularly.",
                  "Prefer gradual changes over extreme diets, detoxes or long fasts."],
    "immunity": ["Prioritise sleep, regular activity and a varied diet.", "Wash hands regularly and ask your clinician about recommended vaccinations."],
    "general": ["Aim for regular sleep, movement, balanced meals and hydration.", "Make one change at a time so you can see what helps."],
}

SEEK_CARE = {
    "digestion": "Seek medical advice if symptoms last more than 2 weeks, keep getting worse, or you notice blood, black stools, weight loss, vomiting that won't stop or trouble swallowing.",
    "hair": "Seek medical advice for sudden or patchy hair loss, a painful, red or scaly scalp, or if shedding continues beyond a few months.",
    "skin": "Seek medical advice for spreading redness, pus, fever, a changing mole or a sore that will not heal, or if there is no improvement after a few weeks.",
    "sleep_stress": "Seek medical advice if problems last more than a few weeks, affect daily life, or come with low mood or panic. Seek urgent help for thoughts of self-harm.",
    "nutrition": "Seek advice from a doctor or registered dietitian for unexplained weight change, persistent fatigue, or any worry about eating habits.",
    "immunity": "Seek medical advice for fever lasting more than a few days, night sweats, swollen glands or repeated infections.",
    "general": "Seek medical advice if symptoms persist, worsen, or you are worried.",
}


@dataclass
class Recommendation:
    kind: str                   # "herb" | "product"
    id: str
    name: str
    category: str
    why: str
    evidence_level: str
    evidence_text: str
    route: str
    how_to_use: str
    precautions: list[str]
    checks: list[str]
    avoid_if: list[str]
    stop_when: list[str]
    seek_care: str
    references: list[str]
    herb_id: str | None = None
    ingredients: list[str] = field(default_factory=list)
    is_demo: bool = False
    verification: str | None = None


@dataclass
class Withheld:
    name: str
    reasons: list[str]


@dataclass
class EngineResult:
    recommendations: list[Recommendation] = field(default_factory=list)
    products: list[Recommendation] = field(default_factory=list)
    withheld: list[Withheld] = field(default_factory=list)
    red_flags: list[sc.RedFlag] = field(default_factory=list)
    needs_info: list[str] = field(default_factory=list)
    lifestyle: list[str] = field(default_factory=list)
    clinician_first: bool = False
    clinician_advice: bool = False
    seek_care: str = ""
    no_product_message: str | None = None

    @property
    def allowed_herb_ids(self) -> set[str]:
        return {r.herb_id for r in self.recommendations if r.herb_id}


def routes_for(profile: UserProfile) -> set[str]:
    pref = (profile.preferences or {}).get("route")
    if pref in ("oral", "topical"):
        return {pref}
    return {"oral", "topical"}


def routes_needed(profile: UserProfile, kb: KnowledgeBase) -> set[str]:
    """Routes the engine could recommend for this concern (decides which safety info is required)."""
    wanted = routes_for(profile)
    cat = profile.concern_category or "general"
    avail = set()
    for herb in kb.candidates_for(cat):
        avail |= set(kb.routes_for(herb, cat))
    return (avail & wanted) or wanted


def _why(herb: dict[str, Any], category: str) -> str:
    studied = [u for u in herb.get("evidence_supported_uses", []) if u.get("category") == category]
    if studied:
        u = studied[0]
        return f"Studied for: {u['use']}. Evidence: {evidence_label(u.get('evidence_level'))}."
    trad = herb.get("traditional_uses") or []
    if trad:
        return f"Traditionally used for: {trad[0].lower()}. There is no reliable clinical proof that it works for this."
    return "Relevant to this area, but reliable information on benefit is limited."


def _refs(herb: dict[str, Any]) -> list[str]:
    out = []
    for s in herb.get("sources", []):
        out.append(f"{s.get('publisher', 'Unknown publisher')}: {s.get('title', '')}" + (f" ({s['url']})" if s.get("url") else " (link not yet verified)"))
    return out


def _avoid_if(herb: dict[str, Any], route: str) -> list[str]:
    items = ["Children and adolescents (this app gives no herbal advice for under-18s)"]
    for key, label in (("pregnancy", "Pregnancy"), ("breastfeeding", "Breastfeeding")):
        status = (herb.get(key) or {}).get(route, "insufficient_data")
        items.append(f"{label}: {'avoid' if status == 'avoid' else 'safety not established, ask your doctor'}")
    conds = herb.get("contraindications", {}).get("conditions", [])
    if conds:
        items.append("Conditions: " + ", ".join(conds))
    kws = herb.get("allergy_info", {}).get("family")
    if kws:
        items.append(f"Allergy to plants in the {kws} family")
    items += herb.get("contraindications", {}).get("notes", [])
    return items


def build_herb_recommendation(herb: dict[str, Any], route: str, category: str, safety: sc.SafetyResult) -> Recommendation:
    precautions = list(dict.fromkeys(herb.get("risks", []) + safety.cautions))
    if route == "topical":
        precautions.append("Patch test on a small area of skin first and wait 24 hours.")
    precautions.append("Try one new product at a time so you can tell what causes a reaction.")
    guidance = herb.get("preparation") or ""
    usage = herb.get("usage_guidance") or ""
    return Recommendation(
        kind="herb", id=f"{herb['id']}:{route}", name=f"{herb['common_name']} ({'on skin/scalp' if route == 'topical' else 'by mouth'})",
        category=category, why=_why(herb, category), evidence_level=herb.get("evidence_level", "unknown"),
        evidence_text=evidence_label(herb.get("evidence_level")), route=route,
        how_to_use=" ".join(x for x in (guidance, usage) if x), precautions=precautions,
        checks=list(dict.fromkeys(safety.checks)), avoid_if=_avoid_if(herb, route),
        stop_when=list(herb.get("when_to_stop", [])), seek_care=SEEK_CARE.get(category, SEEK_CARE["general"]),
        references=_refs(herb), herb_id=herb["id"],
    )


def build_product_recommendation(match: ProductMatch, herb: dict[str, Any] | None, category: str) -> Recommendation:
    p = match.product
    precautions = list(dict.fromkeys(match.safety.cautions + p.get("product_warnings", [])))
    if p.get("route") == "topical":
        precautions.append("Patch test on a small area of skin first and wait 24 hours.")
    return Recommendation(
        kind="product", id=p["id"], name=p["name"], category=category,
        why=p.get("intended_use", ""), evidence_level=(herb or {}).get("evidence_level", "unknown"),
        evidence_text=evidence_label((herb or {}).get("evidence_level")), route=p.get("route", ""),
        how_to_use="Follow the directions on the product label. This app does not state doses.",
        precautions=precautions, checks=list(dict.fromkeys(match.safety.checks)),
        avoid_if=_avoid_if(herb, p.get("route", "")) if herb else [], stop_when=list((herb or {}).get("when_to_stop", [])),
        seek_care=SEEK_CARE.get(category, SEEK_CARE["general"]), references=_refs(herb) if herb else [],
        herb_id=(p.get("herb_ids") or [None])[0], ingredients=list(p.get("ingredients", [])),
        is_demo=bool(p.get("is_demo")), verification=p.get("verification_status"),
    )


def run(profile: UserProfile, kb: KnowledgeBase, catalogue: ProductCatalogue, extra_text: str = "") -> EngineResult:
    """Full evaluation. Safe to call at any time: if information is missing it says so and recommends nothing."""
    result = EngineResult()
    category = profile.concern_category or "general"
    result.seek_care = SEEK_CARE.get(category, SEEK_CARE["general"])

    # Stage 3/4: escalation first
    result.red_flags = sc.check_red_flags(profile, extra_text)
    if any(f.action == "block_herbal" for f in result.red_flags):
        result.clinician_first = True
        return result
    if any(f.action == "advise_clinician" for f in result.red_flags):
        result.clinician_advice = True

    routes = routes_needed(profile, kb)
    result.needs_info = sc.missing_safety_info(profile, "oral" if "oral" in routes else "topical")

    candidates: list[tuple[dict[str, Any], str, sc.SafetyResult]] = []
    for herb in kb.candidates_for(category):
        for route in kb.routes_for(herb, category):
            if route not in routes_for(profile):
                continue
            safety = sc.check_herb(herb, route, profile)
            if safety.allowed:
                candidates.append((herb, route, safety))
            else:
                result.withheld.append(Withheld(f"{herb['common_name']} ({route})", safety.blocks))
    candidates.sort(key=lambda c: (EVIDENCE_RANK.get(c[0].get("evidence_level"), 9), c[0]["common_name"]))

    seen_herbs: set[str] = set()
    for herb, route, safety in candidates:
        if herb["id"] in seen_herbs:
            continue  # one route per herb keeps advice simple
        seen_herbs.add(herb["id"])
        result.recommendations.append(build_herb_recommendation(herb, route, category, safety))
        if len(result.recommendations) >= MAX_HERB_RECOMMENDATIONS:
            break

    # Output must also respect the guard (defence in depth)
    result.recommendations = [r for r in result.recommendations
                              if not sc.scan_output_for_violations(" ".join([r.why, r.how_to_use, *r.precautions]))]

    # Products: only from the catalogue, only if they pass every check
    cand_ids = {h["id"] for h, _, _ in candidates}
    matches, exclusions = match_products(catalogue, kb.herbs, profile, cand_ids, routes_for(profile))
    for m in matches[:3]:
        herb = kb.get((m.product.get("herb_ids") or [""])[0])
        result.products.append(build_product_recommendation(m, herb, category))
    for ex in exclusions:
        result.withheld.append(Withheld(ex.product.get("name", "Unnamed product"), ex.reasons))
    if not result.products:
        result.no_product_message = ("No suitable product from the catalogue passed all safety and preference checks, "
                                     "so I am not suggesting a specific product.")

    if not any(f.action == "block_herbal" for f in result.red_flags):
        result.lifestyle = LIFESTYLE.get(category, LIFESTYLE["general"])
    if category == "nutrition" and any(f.rule_id == "disordered_eating" for f in result.red_flags):
        result.lifestyle = []
    return result


@dataclass
class NamedItemAssessment:
    found: bool
    name: str
    herb: dict[str, Any] | None = None
    verdicts: list[tuple[str, sc.SafetyResult]] = field(default_factory=list)  # (route, result)
    message: str = ""


def assess_named_item(name: str, profile: UserProfile, kb: KnowledgeBase) -> NamedItemAssessment:
    """'Is X suitable for me?' Unknown herbs are never declared safe."""
    herb = kb.find_by_name(name)
    if herb is None or not kb.is_complete(herb["id"]):
        return NamedItemAssessment(
            False, name,
            message=(f"I don't have reliable, reviewed information on '{name}' in my knowledge base, so I can't tell you "
                     "whether it is suitable for you. Please ask a pharmacist or doctor, and bring the product label."))
    out = NamedItemAssessment(True, herb["common_name"], herb)
    for route in herb.get("routes", []):
        out.verdicts.append((route, sc.check_herb(herb, route, profile)))
    return out
