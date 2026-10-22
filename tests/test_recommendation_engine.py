from modules import recommendation_engine as engine
from modules.product_recommender import ProductCatalogue
from modules.session_manager import NONE, PROVIDED, PREG_NEITHER, PREG_PREGNANT, UserProfile


def screened(**kw):
    base = dict(concern_category="digestion", concern_summary="bloating", age=34, sex="female", pregnancy_status=PREG_NEITHER,
                allergies_status=NONE, medications_status=NONE, conditions_status=NONE)
    base.update(kw)
    return UserProfile(**base)


def names(result):
    return {r.herb_id for r in result.recommendations}


def test_screened_adult_gets_recommendations_with_required_fields(kb):
    res = engine.run(screened(), kb, ProductCatalogue.load())
    assert res.recommendations
    for r in res.recommendations:
        assert r.why and r.evidence_text and r.precautions and r.seek_care and r.avoid_if is not None
        assert r.route in ("oral", "topical")


def test_no_more_than_three_single_ingredient_recommendations(kb):
    res = engine.run(screened(), kb, ProductCatalogue.load())
    assert len(res.recommendations) <= engine.MAX_HERB_RECOMMENDATIONS
    assert all(r.herb_id for r in res.recommendations)


# 1. allergy
def test_allergic_ingredient_never_recommended_or_matched_in_products(kb):
    prof = screened(allergies=["peppermint"], allergies_status=PROVIDED)
    res = engine.run(prof, kb, ProductCatalogue.load())
    assert "peppermint" not in names(res)
    assert all("peppermint" not in (p.herb_id or "") for p in res.products)
    assert any("peppermint" in w.name.lower() for w in res.withheld)


def test_lamiaceae_cross_allergy_blocks_family(kb):
    res = engine.run(screened(concern_category="hair", allergies=["mint"], allergies_status=PROVIDED,
                              preferences={"route": "topical"}), kb, ProductCatalogue.load())
    assert "rosemary" not in names(res)


# 3. medication
def test_medication_interaction_removes_flagged_herbs(kb):
    prof = screened(medications=["warfarin"], medications_status=PROVIDED)
    res = engine.run(prof, kb, ProductCatalogue.load())
    assert "ginger" not in names(res)
    assert any("interaction" in " ".join(w.reasons).lower() for w in res.withheld)


# 4. children
def test_child_gets_nothing_but_lifestyle(kb):
    res = engine.run(screened(age=8), kb, ProductCatalogue.load())
    assert not res.recommendations and not res.products
    assert res.lifestyle  # non-herbal guidance only
    for r in res.recommendations:
        assert "dose" not in r.how_to_use.lower()


# 5. pregnancy
def test_pregnant_user_gets_no_oral_herbs(kb):
    res = engine.run(screened(pregnancy_status=PREG_PREGNANT), kb, ProductCatalogue.load())
    assert not [r for r in res.recommendations if r.route == "oral"]
    assert not res.products


# 6. unknown ingredients
def test_unknown_herb_is_not_declared_safe(kb):
    out = engine.assess_named_item("dragon blood tonic", screened(), kb)
    assert not out.found and "can't tell you" in out.message


def test_known_herb_named_item_checked_per_route(kb):
    out = engine.assess_named_item("ginger", screened(medications=["warfarin"], medications_status=PROVIDED), kb)
    assert out.found and all(not res.allowed for _, res in out.verdicts)


def test_incomplete_kb_entry_is_never_recommended():
    from modules.herbal_knowledge_base import KnowledgeBase
    broken = KnowledgeBase.from_entries([{"id": "mystery", "common_name": "Mystery herb", "categories": ["digestion"], "routes": ["oral"]}])
    assert not broken.is_complete("mystery")
    res = engine.run(screened(), broken, ProductCatalogue.load())
    assert not res.recommendations


# 7. products
def test_only_catalogue_products_are_returned(kb):
    cat = ProductCatalogue.load()
    ids = {p["id"] for p in cat.products}
    res = engine.run(screened(), kb, cat)
    assert all(p.id in ids for p in res.products)


def test_empty_catalogue_says_no_product_instead_of_inventing(kb):
    res = engine.run(screened(), kb, ProductCatalogue.from_entries([]))
    assert res.products == [] and res.no_product_message


def test_product_with_allergen_ingredient_is_excluded(kb):
    prof = screened(concern_category="skin", allergies=["shea"], allergies_status=PROVIDED, preferences={"route": "topical"})
    res = engine.run(prof, kb, ProductCatalogue.load())
    assert "DEMO-002" not in {p.id for p in res.products}
    assert any("DEMO Calendula" in w.name for w in res.withheld)


def test_fragrance_free_preference_filters_products(kb):
    prof = screened(concern_category="hair", preferences={"route": "topical", "fragrance_free": True})
    res = engine.run(prof, kb, ProductCatalogue.load())
    assert "DEMO-003" not in {p.id for p in res.products}


def test_demo_products_are_labelled(kb):
    res = engine.run(screened(), kb, ProductCatalogue.load())
    assert res.products and all(p.is_demo for p in res.products)


# escalation beats recommendation
def test_red_flag_blocks_all_recommendations(kb):
    prof = screened(symptom_notes=["there is blood in my stool"])
    res = engine.run(prof, kb, ProductCatalogue.load())
    assert res.clinician_first and not res.recommendations and not res.products and not res.lifestyle


def test_disordered_eating_gets_no_diet_or_herb_advice(kb):
    prof = screened(concern_category="nutrition", symptom_notes=["I've been purging after meals"])
    res = engine.run(prof, kb, ProductCatalogue.load())
    assert res.clinician_first and not res.recommendations


def test_missing_info_withholds_and_reports_needs(kb):
    res = engine.run(UserProfile(concern_category="digestion", age=30, sex="female"), kb, ProductCatalogue.load())
    assert not res.recommendations
    assert {"allergies", "medications"} <= set(res.needs_info)


def test_every_recommendation_has_a_reference_or_explicit_unverified_link(kb):
    res = engine.run(screened(), kb, ProductCatalogue.load())
    assert all(r.references for r in res.recommendations)
