# IT Helpdesk Ticket Triage & SLA Breach Predictor

Automatically classifies incoming IT support tickets by category, flags likely priority based on urgency language, and predicts the probability of SLA breach — deployed as a live, interactive app.

**Live demo:** https://it-ticket-triage-2yfr92qfxjpk2grmhey3dp.streamlit.app

---

## Problem

IT service desks (the kind run by Genpact, EXL, Cognizant, TCS, and similar IT services firms) triage thousands of tickets a day. Two recurring operational questions:

1. What category does this ticket belong to, so it routes to the right team?
2. How likely is this ticket to breach its SLA, so at-risk tickets can be flagged before they blow past deadline?

This project builds both pieces — a text classifier for routing, and a breach-risk model for proactive triage — on top of a real ticket dataset, with a working end-to-end demo.

## Dataset

[IT Service Ticket Classification Dataset](https://www.kaggle.com/datasets/adisongoh/it-service-ticket-classification-dataset) (Kaggle) — 47,837 real IT helpdesk tickets across 8 categories (Hardware, HR Support, Access, Miscellaneous, Storage, Purchase, Internal Project, Administrative rights).

**Note:** the dataset contains ticket text and category only — no priority, timestamp, or resolution-time fields. A priority tier and SLA resolution time were simulated on top of the real ticket text and categories, using category-based triage logic (e.g. Hardware/Access skew toward higher urgency) and a category-level "difficulty" multiplier (e.g. Hardware tickets take proportionally longer, reflecting the need for physical parts). This is disclosed explicitly here and in the code comments — the ticket text and category labels are 100% real; only the SLA layer is synthetic.

## Approach

**1. Ticket classification** (real data)
- Text cleaned and vectorized with TF-IDF (unigrams + bigrams)
- Logistic Regression with class weighting to handle category imbalance
- **Result: 85% accuracy, 0.85 macro F1** across all 8 categories

**2. SLA layer** (simulated, disclosed above)
- Priority (P1/P2/P3) assigned per ticket using category-based triage weights
- SLA targets: P1 = 4h, P2 = 24h, P3 = 72h
- Resolution time simulated with a log-normal distribution, scaled by a category-specific difficulty multiplier, producing a realistic ~22% overall breach rate with meaningful variation by category (Hardware breaches ~34% of the time vs. Access at ~11%)

**3. SLA breach prediction**
- Features: predicted category, priority, SLA target hours, ticket text length
- **Resolution time is deliberately excluded from the feature set** — including it would leak the outcome into the model, since resolution time is only known *after* a ticket is resolved, not at the time a breach-risk prediction would actually be needed
- Gradient Boosting Classifier
- **Result: ROC-AUC 0.65** — modest but real signal. At the default 0.5 classification threshold, the model caught 0% of actual breaches (none of the risk scores exceeded 50%, since even the riskiest ticket combinations only reach ~35% predicted risk). Lowering the threshold to 0.30 improved recall on breached tickets from 0% to 50%, at the cost of lower precision (0.34) — a deliberate tradeoff favoring breach detection over false-alarm avoidance, appropriate for a helpdesk that wants to catch risk early.

**4. Interactive dashboard** (Power BI)
- SLA compliance by category and priority, with a category × priority breach-rate heatmap
- Fully interactive — category/priority slicers cross-filter all visuals live

**5. Deployed app** (Streamlit)
- Paste a raw ticket description → get predicted category, priority (via keyword-based urgency detection with a category-weighted fallback), and SLA breach risk %

## Honest limitations

- **SLA/priority/resolution-time data is simulated**, not real — clearly disclosed above and in code comments. The classification model is trained entirely on real data.
- **Breach signal is weak** (ROC-AUC 0.65) — category, priority, and ticket length alone are limited predictors of breach risk. A production system would need richer features (agent workload, time of day, historical backlog).
- **Some category confusion exists** — e.g., indirectly-phrased Purchase requests are sometimes classified as Miscellaneous.
- **Priority detection uses simple keyword matching** as a first pass (e.g. "down," "critical," "urgent" → P1), falling back to category-based historical proportions when no clear urgency language is present. A production version would use a trained urgency classifier instead of keyword rules.

## Tech stack

Python (pandas, scikit-learn, joblib), SQL (DuckDB), Power BI, Streamlit

## Run it locally

```bash
git clone https://github.com/1rfaan/it-ticket-triage.git
cd it-ticket-triage
pip install -r requirements.txt
streamlit run app.py
```
