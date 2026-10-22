# Herbal Wellness AI

A conversational herbal wellness and nutrition education assistant built with Python and Streamlit.
It asks relevant questions gradually, runs a **deterministic safety engine** (allergies, medicines,
pregnancy, age, warning signs), and only then suggests single-ingredient options from a local knowledge
base, each labelled with how much evidence supports it.

> **Not medical advice.** The app cannot diagnose, treat or replace a doctor or pharmacist. The bundled
> herb data is a **draft starter set that has not been clinically reviewed**, and the product catalogue
> contains **demonstration placeholders only**. Do not use it with real patients until a qualified
> clinician or pharmacist has reviewed the data and rules.

## Features

- **Strict patient intake:** before anything else it asks male/female and age, then concern-specific details (at least four, adjusted for sex, for example menstrual/hormonal questions for women and pattern questions for men), allergies, pregnancy where relevant, medicines, conditions, and daily habits. No suggestion is made until the intake is complete
- Multi-stage conversation: understand concern, find missing information, safety screening, escalation check, recommendations, follow-up
- Gradual health profile held in session memory (no forced form), editable in the sidebar, empty fields never shown as answered
- Concern-specific questions (hair, skin, digestion, sleep/stress, nutrition, immunity, general); answered questions are not repeated
- Deterministic safety layer, run before recommending and again before display: emergency detection (with negation handling), concern-specific red flags, allergy and cross-reactivity checks, medication interaction classes, condition contraindications, age and pregnancy/breastfeeding restrictions
- "Unknown" is never "safe": missing data (unknown medicine, unanswered allergy question, herb not in the database) blocks the recommendation and says why
- Herb knowledge base (`data/herbs.json`, 15 herbs) with evidence levels, risks, interactions and sources
- Product catalogue (`data/products.json`) that can only return listed products, screened ingredient by ingredient
- Optional AI layer (Anthropic SDK) that only *words* replies; its output is validated (JSON schema, approved herbs only, no doses/cures/diagnoses, HTML and links stripped) and falls back to guided templates on any problem
- Works without an API key in clearly labelled **guided mode**

## Herbalist mode (default)

The chatbot now works like a professional herbalist: describe a concern and you get a practical plan in the first reply (herbs, kitchen remedies, how to prepare them, amounts, how long, when to see a doctor) instead of a long intake.

- **Graded safety, not blanket refusals.** Each herb is screened for *this* patient (medicine classes, conditions, allergies, age, pregnancy) as ok / note / caution / avoid. Unfamiliar medicines and significant conditions change *which* herb and *how much*; they no longer end the consultation. Homeopathic remedies and ordinary vitamins are treated as non-interacting.
- **Triage.** Emergency wording stops everything. Warning signs (blood, unexplained weight loss, symptoms lasting too long, etc.) produce a doctor referral with comfort measures only.
- **How the "training" works.** The AI is not fine-tuned. It is guided by a professional-herbalist system prompt (`modules/herbalist_prompt.py`), a compiled knowledge base (`data/herbalist_herbs.json` 58 herbs, `data/herbalist_conditions.json` 35 minor-condition protocols, `data/herbalist_rules.json`), worked examples (`data/herbalist_examples.json`), and code guardrails that reject unsafe or blanket-refusal replies. If the AI is unavailable or its reply fails the guardrails, a built-in deterministic herbalist answers.
- **Switch back** to the strict intake flow with `HERBALIST_MODE=0`.
- **Evals:** `python evals/run_evals.py` checks the behaviour on sample patients.
- **Review needed.** The herb data was compiled from general herbal literature and has not had clinical review. Have a qualified herbalist or pharmacist check amounts and interactions before real-patient use.

## How it works

```
user message
   -> safety_checker.scan_for_emergency            (emergency? stop, show escalation)
   -> health_assessment.apply_user_message         (update profile, no repeats)
   -> safety_checker.check_red_flags               (warning signs? stop, advise a clinician)
   -> health_assessment.is_ready / next_questions  (ask only what matters)
   -> recommendation_engine.run                    (check_herb / check_product for every candidate)
   -> response_validator.final_safety_pass         (second independent screen)
   -> ai_client + response_validator (optional)    (wording only, validated)
   -> app.py renders cards, precautions, held-back options
```

## Project structure

```
herbal_wellness_ai/
├── app.py                       Streamlit UI
├── config.py                    settings and secrets lookup
├── requirements.txt  .env.example  .gitignore
├── .streamlit/config.toml       theme
├── data/
│   ├── herbs.json               herb knowledge base (draft, needs clinical review)
│   ├── products.json            product catalogue (demo placeholders)
│   └── safety_rules.json        emergency/red-flag patterns, medication classes, allergen synonyms
├── modules/
│   ├── ai_client.py             provider abstraction + Anthropic implementation
│   ├── conversation_manager.py  one conversational turn, stages
│   ├── health_assessment.py     question bank, extraction, readiness
│   ├── herbal_knowledge_base.py load and validate herbs
│   ├── product_recommender.py   catalogue loading and safe matching
│   ├── recommendation_engine.py ranking, cards, lifestyle, named-item checks
│   ├── response_validator.py    model-output validation, final safety pass
│   ├── safety_checker.py        all deterministic safety rules
│   └── session_manager.py       profile and session state, reset
├── utils/  validators.py  formatting.py
└── tests/                       pytest suite (see below)
```

## Requirements

Python 3.10 or newer (developed on 3.13).

## Installation (Windows PowerShell)

```powershell
cd herbal_wellness_ai
python -m venv venv
.\venv\Scripts\Activate.ps1
# If activation is blocked:  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
pip install -r requirements.txt
```

macOS/Linux: `python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`

## Configure the API key (optional)

Without a key the app runs in guided mode. For AI-worded replies:

```powershell
Copy-Item .env.example .env
notepad .env        # set ANTHROPIC_API_KEY=your real key
```

Or set it for the current PowerShell session only: `$env:ANTHROPIC_API_KEY = "your-key"`.
You can also use Streamlit secrets (`.streamlit/secrets.toml`, already git-ignored):
`ANTHROPIC_API_KEY = "your-key"`. Optional: `ANTHROPIC_MODEL`, `AI_TIMEOUT_SECONDS`, `AI_MAX_TOKENS`.
Never commit `.env`. Get a key at https://console.anthropic.com/.

## Run

```powershell
streamlit run app.py
```

## Test

```powershell
python -m pytest tests -q
```

## Add or edit data

**Add a herb** - append an object to `data/herbs.json` using an existing entry as the template. Required fields are
listed in `modules/herbal_knowledge_base.py` (`REQUIRED_FIELDS`); an entry missing any of them is automatically
excluded from recommendations. Use `"insufficient_data"` for unknown pregnancy/breastfeeding safety (treated as *not safe*).
Add `category_routes` when the sensible route differs by concern (e.g. turmeric: oral for digestion, topical for skin).
Never invent evidence, doses or citations; add real sources and set `review_status` only after clinical review.

**Add a product** - append to `data/products.json` with a full ingredient list copied from the label, `herb_ids` that
exist in `herbs.json`, `route`, `official_url`, `source_of_information` and `last_verified`. Remove `is_demo`
only for verified real products. Never add a product without a traceable source.

**Update safety rules** - edit `data/safety_rules.json` (emergency and red-flag regex patterns, medication classes,
condition keywords, allergen synonyms), then run the tests. If you add a herb interaction, use a `medication_class`
that exists under `medication_classes`.

## Privacy

- Chat and health profile live only in Streamlit session memory; nothing is written to a database or log by this app
- "Clear my data now" / "New conversation" wipes the session; closing the browser tab ends it
- AI is **off by default** even when a key exists: a sidebar toggle must be switched on before any text is sent to the provider
- When on, only the last ~12 messages and the non-empty profile fields are sent (no names or contact details are ever requested)
- Session memory is not a compliance claim: this app has not been assessed against HIPAA, GDPR or any other law

## Known limitations

- The herb data is a small, **unreviewed draft** and the product catalogue is fake demonstration data
- Recommendations match on concern category and evidence level; they do not yet tailor to the specific symptom (for example psyllium for constipation versus bloating)
- Interaction checking is class-based and incomplete by design; unknown medicines block oral herbs rather than being assumed safe
- Free-text understanding is rule-based: unusual phrasing may be missed, so ambiguous replies are re-asked (up to 3 times) and then recorded as "not sure", never as safe
- A concern-specific screening question counts as answered once the user replies; the reply is not clinically interpreted
- Emergency and red-flag detection is keyword-based, handles simple negation only, and is not a triage system
- English only; no persistence across sessions; no user accounts
- The AI path has been tested with a fake provider only; it has not been run against the live Anthropic API in this build

## Deployment considerations

- Keep `.env` out of source control; use your host's secret store (or Streamlit secrets)
- Serve over HTTPS; add authentication if exposed publicly
- Do not enable request logging that captures chat text
- Clinical review of herbs, rules and wording by a qualified professional, plus legal review for your region, is required before any real-world launch
