

import pandas as pd 
df = pd.read_csv("C:/Users/peeye/Desktop/MyProjects/it-ticket-triage/data/raw/all_tickets_processed_improved_v3.csv")  
print(df.shape)
print(df.columns.tolist())
df.head()

print(df['Topic_group'].value_counts())
print(df['Topic_group'].nunique())
df['doc_length'] = df['Document'].str.split().str.len()
print(df['doc_length'].describe())





#Clean the text and set up a proper train/test split

import re

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)   # strip punctuation/special chars
    text = re.sub(r'\s+', ' ', text).strip()    # collapse whitespace
    return text

df['clean_text'] = df['Document'].apply(clean_text)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['Topic_group'],
    test_size=0.2, random_state=42, stratify=df['Topic_group']
)

print(X_train.shape, X_test.shape)
print(y_train.value_counts(normalize=True))








# TF-IDF + Logistic Regression baseline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
clf.fit(X_train_tfidf, y_train)

y_pred = clf.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))





#Save the model and vectorizer, then move to the SLA simulation

import joblib

joblib.dump(clf, 'src/ticket_classifier.pkl')
joblib.dump(vectorizer, 'src/tfidf_vectorizer.pkl')









#  Simulate priority (category-based) and resolution time

import numpy as np

np.random.seed(42)

# Priority tendency per category — based on typical IT helpdesk triage logic:
# Hardware/Access = blocking issues -> skew urgent
# HR Support/Purchase/Internal Project = rarely urgent -> skew low
priority_weights = {
    'Hardware':               [0.35, 0.45, 0.20],  # [P1, P2, P3]
    'Access':                 [0.30, 0.50, 0.20],
    'Storage':                [0.25, 0.45, 0.30],
    'Miscellaneous':          [0.10, 0.40, 0.50],
    'Administrative rights':  [0.15, 0.40, 0.45],
    'HR Support':             [0.05, 0.25, 0.70],
    'Purchase':                [0.05, 0.25, 0.70],
    'Internal Project':       [0.05, 0.20, 0.75],
}

def assign_priority(category):
    weights = priority_weights[category]
    return np.random.choice(['P1', 'P2', 'P3'], p=weights)

df['priority'] = df['Topic_group'].apply(assign_priority)

# SLA targets (in hours) per priority tier — a common industry convention
sla_hours = {'P1': 4, 'P2': 24, 'P3': 72}
df['sla_target_hours'] = df['priority'].map(sla_hours)

# Simulate resolution time: drawn from a distribution centered around the SLA,
# so some tickets naturally breach it (realistic — not every ticket meets SLA)
category_difficulty = {
    'Hardware':               1.3,
    'Purchase':                1.25,
    'Internal Project':       1.2,
    'Storage':                1.1,
    'Administrative rights':  1.0,
    'Miscellaneous':          1.0,
    'Access':                 0.8,
    'HR Support':             0.85,
}

def simulate_resolution_time(priority, category):
    target = sla_hours[priority]
    difficulty = category_difficulty[category]
    return np.random.lognormal(mean=np.log(target * 0.6 * difficulty), sigma=0.6)

df['resolution_hours'] = df.apply(
    lambda row: simulate_resolution_time(row['priority'], row['Topic_group']), axis=1
)
df['sla_breached'] = (df['resolution_hours'] > df['sla_target_hours']).astype(int)

print(df[['priority', 'sla_target_hours', 'resolution_hours', 'sla_breached']].head(10))
print(df['sla_breached'].value_counts(normalize=True))
print(df.groupby('priority')['sla_breached'].mean())









#Save the enriched dataset

df.to_csv('data/processed/tickets_enriched.csv', index=False)





#Track A: the SLA breach prediction model.
from sklearn.preprocessing import LabelEncoder

# Features: what we'd realistically know at ticket creation time
# (NOT resolution_hours — that's only known AFTER the ticket is resolved,
# so including it would be leaking the answer into the model)
features_df = df[['Topic_group', 'priority', 'sla_target_hours', 'doc_length']].copy()

le_category = LabelEncoder()
le_priority = LabelEncoder()

features_df['category_encoded'] = le_category.fit_transform(features_df['Topic_group'])
features_df['priority_encoded'] = le_priority.fit_transform(features_df['priority'])

X = features_df[['category_encoded', 'priority_encoded', 'sla_target_hours', 'doc_length']]
y = df['sla_breached']

print(X.head())
print(y.value_counts(normalize=True))






# Train/test split and train the breach classifier

X_train_sla, X_test_sla, y_train_sla, y_test_sla = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

from sklearn.ensemble import GradientBoostingClassifier

sla_clf = GradientBoostingClassifier(random_state=42)
sla_clf.fit(X_train_sla, y_train_sla)

y_pred_sla = sla_clf.predict(X_test_sla)
y_proba_sla = sla_clf.predict_proba(X_test_sla)[:, 1]

print(classification_report(y_test_sla, y_pred_sla))









#Fix: look at the actual probability distribution, then pick a better threshold.
import numpy as np
from sklearn.metrics import roc_auc_score

print("Lowest risk score:", y_proba_sla.min())
print("Highest risk score:", y_proba_sla.max())
print("Risk score at different points (50th/75th/90th/95th percentile):")
print(np.percentile(y_proba_sla, [50, 75, 90, 95, 99]))

print("ROC-AUC score:", roc_auc_score(y_test_sla, y_proba_sla))


#Picks a smarter threshold and re-evaluate
threshold = 0.30
y_pred_adjusted = (y_proba_sla >= threshold).astype(int)

print(classification_report(y_test_sla, y_pred_adjusted))



#Save the SLA breach model and its encoders
joblib.dump(sla_clf, 'src/sla_breach_model.pkl')
joblib.dump(le_category, 'src/category_encoder.pkl')
joblib.dump(le_priority, 'src/priority_encoder.pkl')