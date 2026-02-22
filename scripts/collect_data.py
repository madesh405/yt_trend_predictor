import os
import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_auc_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# -------------------------------------------------
# Paths
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "trendpulse_channel_relative.csv")
models_dir = os.path.join(BASE_DIR, "models")
os.makedirs(models_dir, exist_ok=True)


# -------------------------------------------------
# Load Dataset
# -------------------------------------------------
df = pd.read_csv(data_path)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nDataset Size:", len(df))


# =================================================
# 🔥 NLP FEATURE ENGINEERING
# =================================================
analyzer = SentimentIntensityAnalyzer()

def extract_nlp_features(text):
    blob = TextBlob(text)

    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    vader_score = analyzer.polarity_scores(text)["compound"]

    exclamations = text.count("!")
    questions = text.count("?")

    word_count = len(text.split())
    unique_words = len(set(text.split()))
    lexical_diversity = unique_words / (word_count + 1)

    return pd.Series([
        polarity,
        subjectivity,
        vader_score,
        exclamations,
        questions,
        lexical_diversity
    ])


df[[
    "polarity",
    "subjectivity",
    "vader_score",
    "exclamations",
    "questions",
    "lexical_diversity"
]] = df["full_text"].fillna("").apply(extract_nlp_features)


# =================================================
# Combine With Numeric Features
# =================================================
NUMERIC_COLUMNS = [
    # Original
    "title_length",
    "caps_ratio",
    "duration_sec",
    "publish_hour",
    "subscriber_count",
    "views_per_video",

    # Engagement
    "like_ratio",
    "comment_ratio",
    "velocity",

    # NLP
    "polarity",
    "subjectivity",
    "vader_score",
    "exclamations",
    "questions",
    "lexical_diversity"
]

X = df[NUMERIC_COLUMNS].fillna(0)
y = df["viral"]
groups = df["channel_id"]


# -------------------------------------------------
# Scale
# -------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# -------------------------------------------------
# Channel-wise Split
# -------------------------------------------------
gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
train_idx, test_idx = next(gss.split(X_scaled, y, groups))

X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]


# =================================================
# Logistic Regression
# =================================================
model = LogisticRegression(max_iter=2000, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n=== RESULTS WITH NLP + ENGAGEMENT ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# -------------------------------------------------
# Save
# -------------------------------------------------
joblib.dump(model, os.path.join(models_dir, "early_model.pkl"))
joblib.dump(scaler, os.path.join(models_dir, "early_scaler.pkl"))

print("\nEnhanced Model Saved Successfully.")