import streamlit as st
import joblib
import re
import numpy as np

# Load all saved models/encoders
ticket_classifier = joblib.load('src/ticket_classifier.pkl')
tfidf_vectorizer = joblib.load('src/tfidf_vectorizer.pkl')
sla_model = joblib.load('src/sla_breach_model.pkl')
category_encoder = joblib.load('src/category_encoder.pkl')
priority_encoder = joblib.load('src/priority_encoder.pkl')

# Priority weights - same logic as your simulation, used to assign a
# realistic priority to new tickets based on predicted category
priority_weights = {
    'Hardware':               [0.35, 0.45, 0.20],
    'Access':                 [0.30, 0.50, 0.20],
    'Storage':                [0.25, 0.45, 0.30],
    'Miscellaneous':          [0.10, 0.40, 0.50],
    'Administrative rights':  [0.15, 0.40, 0.45],
    'HR Support':             [0.05, 0.25, 0.70],
    'Purchase':                [0.05, 0.25, 0.70],
    'Internal Project':       [0.05, 0.20, 0.75],
}
sla_hours = {'P1': 4, 'P2': 24, 'P3': 72}

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

st.title("IT Helpdesk Ticket Triage")
st.write("Paste a ticket description below to predict its category and SLA breach risk.")

ticket_text = st.text_area("Ticket description", height=120)

if st.button("Predict"):
    if ticket_text.strip() == "":
        st.warning("Please enter a ticket description.")
    else:
        # Step 1: Predict category
        cleaned = clean_text(ticket_text)
        text_tfidf = tfidf_vectorizer.transform([cleaned])
        predicted_category = ticket_classifier.predict(text_tfidf)[0]

        # Step 2: Detect urgency from ticket text, then assign priority
        urgent_keywords = ['down', 'critical', 'urgent', 'production', 'outage',
                             'can\'t access', 'cannot access', 'blocked', 'emergency',
                             'not working', 'crashed', 'broken', 'asap']
        routine_keywords = ['requesting', 'would like', 'please add', 'new',
                              'when possible', 'update my']

        text_lower = cleaned.lower()
        urgency_score = sum(1 for kw in urgent_keywords if kw in text_lower)
        routine_score = sum(1 for kw in routine_keywords if kw in text_lower)

        if urgency_score >= 1:
            predicted_priority = 'P1'
        elif routine_score >= 1:
            predicted_priority = 'P3'
        else:
            # No clear signal in text — fall back to category's typical priority distribution
            weights = priority_weights[predicted_category]
            predicted_priority = np.random.choice(['P1', 'P2', 'P3'], p=weights)
        sla_target = sla_hours[predicted_priority]
        doc_len = len(cleaned.split())

        # Step 3: Predict breach risk
        cat_encoded = category_encoder.transform([predicted_category])[0]
        pri_encoded = priority_encoder.transform([predicted_priority])[0]
        features = np.array([[cat_encoded, pri_encoded, sla_target, doc_len]])
        breach_risk = sla_model.predict_proba(features)[0][1]

        # Display results
        st.subheader("Results")
        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted Category", predicted_category)
        col2.metric("Likely Priority", predicted_priority)
        col3.metric("SLA Breach Risk", f"{breach_risk:.0%}")

        if breach_risk >= 0.30:
            st.error(f"High risk — flag for extra attention (threshold: 30%)")
        else:
            st.success("Low risk — standard handling")