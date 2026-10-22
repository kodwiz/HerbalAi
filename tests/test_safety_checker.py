from modules import safety_checker as sc
from modules.session_manager import NONE, PROVIDED, PREG_BREASTFEEDING, PREG_PREGNANT


# 1. Known allergy to proposed ingredient -> never recommended
def test_allergy_blocks_herb(kb, adult):
    adult.allergies, adult.allergies_status = ["ragweed"], PROVIDED
    r = sc.check_herb(kb.get("chamomile"), "oral", adult)
    assert not r.allowed and any("Allergy" in b for b in r.blocks)


def test_direct_allergy_blocks_herb(kb, adult):
    adult.allergies, adult.allergies_status = ["ginger"], PROVIDED
    assert not sc.check_herb(kb.get("ginger"), "oral", adult).allowed


def test_peanut_allergy_blocks_fenugreek(kb, adult):
    adult.allergies, adult.allergies_status = ["peanuts"], PROVIDED
    assert not sc.check_herb(kb.get("fenugreek"), "oral", adult).allowed


def test_tree_nut_allergy_blocks_shea_product(kb, adult):
    adult.concern_category = "skin"
    adult.allergies, adult.allergies_status = ["tree nuts"], PROVIDED
    product = {"id": "X", "route": "topical", "herb_ids": ["calendula"], "ingredients": ["calendula extract", "shea butter"]}
    r = sc.check_product(product, kb.herbs, adult)
    assert not r.allowed and any("ingredient" in b.lower() for b in r.blocks)


def test_unknown_allergy_status_blocks_oral(kb, adult):
    adult.allergies_status = "prefer_not"
    r = sc.check_herb(kb.get("ginger"), "oral", adult)
    assert not r.allowed and "allergies" in r.missing_info


def test_clean_adult_can_get_low_risk_herb(kb, adult):
    r = sc.check_herb(kb.get("peppermint"), "oral", adult)
    assert r.allowed, r.blocks


# 2. Emergencies
def test_emergency_chest_pain():
    hits = sc.scan_for_emergency("I have chest pain and my left arm is numb")
    assert hits and hits[0].rule_id == "chest_pain"
    assert "emergency" in sc.emergency_response(hits).lower()


def test_emergency_breathing_and_swelling():
    assert sc.scan_for_emergency("my lips are swelling and I can't breathe properly")


def test_emergency_stroke_signs():
    assert sc.scan_for_emergency("sudden weakness on one side and slurred speech")


def test_negated_emergency_is_ignored():
    assert sc.scan_for_emergency("I have bloating but no chest pain") == []


def test_crisis_message_has_no_herbal_advice():
    hits = sc.scan_for_emergency("I want to die")
    assert hits and hits[0].is_crisis
    assert "herb" in sc.emergency_response(hits).lower()  # only to say it is not offering herbal advice
    assert "recommend" not in sc.emergency_response(hits).lower()


def test_no_false_alarm_on_ordinary_text():
    assert sc.scan_for_emergency("my hair has been falling out for three months") == []


# 3. Medication interactions
def test_anticoagulant_blocks_ginger(kb, adult):
    adult.medications, adult.medications_status = ["warfarin 5mg"], PROVIDED
    r = sc.check_herb(kb.get("ginger"), "oral", adult)
    assert not r.allowed and r.interaction_status == "flagged"


def test_thyroid_med_blocks_ashwagandha(kb, adult):
    adult.medications, adult.medications_status = ["levothyroxine"], PROVIDED
    assert not sc.check_herb(kb.get("ashwagandha"), "oral", adult).allowed


def test_unknown_medication_gives_insufficient_not_safe(kb, adult):
    adult.medications, adult.medications_status = ["zorblaxin"], PROVIDED
    r = sc.check_herb(kb.get("peppermint"), "oral", adult)
    assert not r.allowed and r.interaction_status == "insufficient"


def test_medications_not_shared_blocks_oral(kb, adult):
    adult.medications_status = "unanswered"
    r = sc.check_herb(kb.get("peppermint"), "oral", adult)
    assert not r.allowed and "medications" in r.missing_info


def test_medication_classification():
    classes, status = sc.classify_medication("Warfarin 3 mg")
    assert "anticoagulants" in classes and status == "classified"
    assert sc.classify_medication("zorblaxin")[1] == "unknown"


def test_licorice_is_restricted(kb, adult):
    assert not sc.check_herb(kb.get("licorice"), "oral", adult).allowed


# 4. Children
def test_child_gets_no_herbal_recommendation(kb, adult):
    adult.age = 9
    for herb_id in ("peppermint", "ginger", "aloe_vera"):
        route = kb.get(herb_id)["routes"][0]
        r = sc.check_herb(kb.get(herb_id), route, adult)
        assert not r.allowed


def test_adolescent_blocked(kb, adult):
    adult.age = 16
    assert not sc.check_herb(kb.get("peppermint"), "oral", adult).allowed


def test_age_unknown_blocks(kb, adult):
    adult.age = None
    r = sc.check_herb(kb.get("peppermint"), "oral", adult)
    assert not r.allowed and "age" in r.missing_info


# 5. Pregnancy
def test_pregnant_oral_blocked(kb, adult):
    adult.pregnancy_status = PREG_PREGNANT
    for hid in ("ginger", "peppermint", "chamomile", "psyllium_husk"):
        assert not sc.check_herb(kb.get(hid), "oral", adult).allowed


def test_breastfeeding_blocked_for_unverified(kb, adult):
    adult.pregnancy_status = PREG_BREASTFEEDING
    assert not sc.check_herb(kb.get("ginger"), "oral", adult).allowed


def test_pregnant_topical_unverified_blocked(kb, adult):
    adult.pregnancy_status = PREG_PREGNANT
    assert not sc.check_herb(kb.get("aloe_vera"), "topical", adult).allowed


def test_pregnancy_unknown_for_woman_blocks(kb, adult):
    adult.pregnancy_status = "unknown"
    r = sc.check_herb(kb.get("peppermint"), "oral", adult)
    assert not r.allowed and "pregnancy" in r.missing_info


# conditions
def test_reflux_blocks_peppermint(kb, adult):
    adult.conditions, adult.conditions_status = ["acid reflux"], PROVIDED
    assert not sc.check_herb(kb.get("peppermint"), "oral", adult).allowed


def test_serious_condition_blocks_oral(kb, adult):
    adult.conditions, adult.conditions_status = ["kidney disease"], PROVIDED
    assert not sc.check_herb(kb.get("ginger"), "oral", adult).allowed


# red flags
def test_hair_patchy_loss_flag(adult):
    adult.concern_category = "hair"
    adult.symptom_notes = ["I have bald patches on the side"]
    flags = sc.check_red_flags(adult)
    assert any(f.action == "block_herbal" for f in flags)


def test_diffuse_hair_loss_has_no_blocking_flag(adult):
    adult.concern_category = "hair"
    adult.symptom_notes = ["hair falling out evenly for three months"]
    assert not [f for f in sc.check_red_flags(adult) if f.action == "block_herbal"]


def test_blood_in_stool_flag(adult):
    adult.concern_category = "digestion"
    adult.symptom_notes = ["there is blood in my stool"]
    assert any(f.rule_id == "gi_bleeding" for f in sc.check_red_flags(adult))


def test_persistent_digestive_symptoms_advise(adult):
    adult.concern_category = "digestion"
    adult.duration_days = 30
    assert any(f.action == "advise_clinician" for f in sc.check_red_flags(adult))


def test_disordered_eating_flag(adult):
    adult.concern_category = "nutrition"
    adult.symptom_notes = ["I have been purging after meals"]
    assert any(f.rule_id == "disordered_eating" for f in sc.check_red_flags(adult))


# output guard
def test_output_guard_catches_cure_and_dose():
    assert sc.scan_output_for_violations("This herb will cure your acne")
    assert sc.scan_output_for_violations("Take 500 mg per day of ginger")
    assert sc.scan_output_for_violations("You should stop your medication and use turmeric")
    assert sc.scan_output_for_violations("You have diabetes")


def test_output_guard_allows_normal_text():
    assert sc.scan_output_for_violations("Peppermint tea is traditionally used after meals. Evidence is limited.") == []
