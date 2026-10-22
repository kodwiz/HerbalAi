import json

from modules import response_validator as rv


def ok_json(msg="Peppermint tea is traditionally used after meals.", herbs=("peppermint",)):
    return json.dumps({"message": msg, "mentioned_herbs": list(herbs)})


# 9. malformed AI output
def test_non_json_rejected(kb):
    assert not rv.validate_model_response("Sure! Take ginger.", {"ginger"}, kb).ok


def test_truncated_json_rejected(kb):
    assert not rv.validate_model_response('{"message": "Hello', set(), kb).ok


def test_wrong_types_rejected(kb):
    assert not rv.validate_model_response('{"message": 42, "mentioned_herbs": []}', set(), kb).ok
    assert not rv.validate_model_response('{"message": "hi", "mentioned_herbs": "ginger"}', set(), kb).ok
    assert not rv.validate_model_response("[1,2,3]", set(), kb).ok
    assert not rv.validate_model_response(None, set(), kb).ok
    assert not rv.validate_model_response('{"message": "  "}', set(), kb).ok


def test_valid_response_accepted_and_code_fences_tolerated(kb):
    r = rv.validate_model_response("```json\n" + ok_json() + "\n```", {"peppermint"}, kb)
    assert r.ok and "Peppermint" in r.message


def test_unapproved_herb_in_text_rejected(kb):
    r = rv.validate_model_response(ok_json("Ginger may help too.", ()), {"peppermint"}, kb)
    assert not r.ok


def test_unapproved_herb_in_mentioned_list_rejected(kb):
    assert not rv.validate_model_response(ok_json("Try this.", ("ginger",)), {"peppermint"}, kb).ok


def test_herb_not_in_knowledge_base_rejected(kb):
    assert not rv.validate_model_response(ok_json("Ashwagandha is nice.", ("unicorn_root",)), {"ashwagandha"}, kb).ok
    assert not rv.validate_model_response(ok_json("Valerian helps sleep.", ()), set(), kb).ok


def test_unsafe_claims_rejected(kb):
    for msg in ("This will cure your acne.", "Take 500 mg per day.", "You should stop your medication.", "You have thyroid disease."):
        assert not rv.validate_model_response(ok_json(msg, ()), set(), kb).ok, msg


def test_html_and_links_are_stripped(kb):
    r = rv.validate_model_response(ok_json("<script>alert(1)</script>Hello [click](http://evil.example) and http://x.example", ()), set(), kb)
    assert r.ok and "<" not in r.message and "http" not in r.message and "evil" not in r.message


def test_overlong_message_is_truncated(kb):
    r = rv.validate_model_response(ok_json("word " * 1000, ()), set(), kb)
    assert r.ok and len(r.message) <= rv.MAX_MESSAGE_CHARS + 2


def test_final_safety_pass_drops_recommendation_that_became_unsafe(kb):
    from modules import recommendation_engine as engine
    from modules.product_recommender import ProductCatalogue
    from modules.session_manager import NONE, PREG_NEITHER, UserProfile
    prof = UserProfile(concern_category="digestion", age=30, sex="female", pregnancy_status=PREG_NEITHER,
                       allergies_status=NONE, medications_status=NONE, conditions_status=NONE)
    cat = ProductCatalogue.load()
    res = engine.run(prof, kb, cat)
    assert res.recommendations
    prof.allergies, prof.allergies_status = ["ginger", "peppermint", "psyllium"], "provided"   # profile changed after the run
    fp = rv.final_safety_pass(res, prof, kb, cat)
    assert fp.dropped and not any(r.herb_id in ("ginger", "peppermint", "psyllium_husk") for r in res.recommendations)
