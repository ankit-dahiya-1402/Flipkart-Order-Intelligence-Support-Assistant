# Flipkart Order Intelligence & Support Assistant

Student capstone project: one connected app combining three components for
e-commerce operations:

1. **Return-Risk Prediction** (`return_risk.py`) — predicts whether an order
   will be returned, with a risk bucket and plain-language reasons.
2. **Catalog / Image Intelligence** (`catalog_model.py`) — classifies a
   product image and flags mismatches against its declared catalog category.
3. **AI Support Assistant** (`assistant.py`) — answers policy questions via
   retrieval (RAG), can call components 1 and 2 as tools, blocks prompt
   injection, and refuses/escalates ungrounded questions.
4. **Streamlit app** (`app.py`) — a 3-tab demo UI over all three, importing
   the modules directly (no API layer, no database — kept intentionally
   simple for a course project).

## Important data disclaimers

- **Orders are synthetic.** Real Flipkart order data was not available, so
  `data/generate_orders.py` generates a synthetic dataset with realistic,
  multi-feature structure (the return label depends on ~8 features, not one).
- **Product images are Fashion-MNIST**, a public apparel image dataset, used
  only as a stand-in for real Flipkart catalog photos because real ones were
  not available. `data/build_catalog.py` deliberately mislabels ~25% of a
  sample so there are real mismatches to demo. This is clearly **not** real
  Flipkart catalog data — the app says so in its header.
- **support_policy.md** is a short, original sample policy document written
  for this project, not Flipkart's actual policy.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Optional: copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY` to get
live LLM-generated assistant replies. Without it, the assistant uses a
deterministic fallback (still fully functional, just less fluent).

## Build the data and models (run once, in order)

```bash
python data/generate_orders.py     # -> data/orders.csv
python data/build_catalog.py       # -> data/catalog.csv, data/sample_images/
python return_risk.py              # -> models/return_risk_model.pkl
python catalog_model.py            # -> models/catalog_classifier.pt
```

## Run the demo

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501) and try:
- **Return Risk tab**: pick a sample order, click Predict.
- **Catalog Check tab**: pick a sample product, click Check category.
- **Support Chat tab**: ask a policy question ("how long to return apparel?"),
  a routed question ("return risk for order 12"), a catalog question
  ("check P0000"), or try an injection attempt to see it get blocked.

## What was verified

- `return_risk.py`: trained Dummy / Logistic Regression / Random Forest,
  compared metrics, tuned an F1-optimal decision threshold, computed
  permutation feature importance, and produces per-order explanations.
- `catalog_model.py`: fine-tuned a ResNet-18 head on a Fashion-MNIST subset
  (~76% test accuracy) and correctly flags a known category match/mismatch.
- `assistant.py`: verified policy retrieval, prompt-injection blocking,
  ungrounded-question refusal + escalation, and both tool-routed intents
  (return-risk lookup, catalog check), with the deterministic fallback
  (no API key was configured in this environment).
- `app.py`: Streamlit starts and serves (HTTP 200) with all three tabs.

## Known limitations (by design, for a study project)

- No database, API layer, authentication, or deployment config — the
  Streamlit app imports the Python modules directly.
- The catalog classifier is trained on a small subset for a few epochs to
  keep training fast; accuracy is good-enough-for-demo, not production-grade.
- The assistant's retrieval is a small in-memory cosine-similarity search
  (sentence-transformers embeddings), sufficient for the dozen or so policy
  chunks here — not a production vector database.
