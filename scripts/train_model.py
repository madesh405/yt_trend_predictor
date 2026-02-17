import os
import pandas as pd
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack


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
print("\nClass Distribution:")
print(df["viral"].value_counts())


# -------------------------------------------------
# Define Features
# -------------------------------------------------
TEXT_COLUMN = "full_text"

NUMERIC_COLUMNS = [
    "title_length",
    "caps_ratio",
    "like_ratio",
    "comment_ratio",
    "velocity",
    "subscriber_count",
    "views_per_video",
    "duration_sec",
    "publish_hour"
]

X_text = df[TEXT_COLUMN].fillna("")
X_numeric = df[NUMERIC_COLUMNS].fillna(0)

y = df["viral"]


# -------------------------------------------------
# TF-IDF Vectorization
# -------------------------------------------------
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=8000,
    ngram_range=(1, 2),
    min_df=2
)

X_text_vec = vectorizer.fit_transform(X_text)


# -------------------------------------------------
# Scale Numeric Features
# -------------------------------------------------
scaler = StandardScaler()
X_numeric_scaled = scaler.fit_transform(X_numeric)

# Convert to sparse matrix to combine with TF-IDF
from scipy.sparse import csr_matrix
X_numeric_sparse = csr_matrix(X_numeric_scaled)


# -------------------------------------------------
# Combine Text + Numeric
# -------------------------------------------------
X_final = hstack([X_text_vec, X_numeric_sparse])


# -------------------------------------------------
# Train/Test Split
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_final,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -------------------------------------------------
# Train Model
# -------------------------------------------------
model = LogisticRegression(
    max_iter=4000,
    class_weight="balanced",
    C=1.2
)

model.fit(X_train, y_train)


# -------------------------------------------------
# Evaluate
# -------------------------------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nTest Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# -------------------------------------------------
# Cross Validation
# -------------------------------------------------
cv_scores = cross_val_score(model, X_final, y, cv=5)

print("\nCross Validation Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())


# -------------------------------------------------
# Save Everything
# -------------------------------------------------
model_path = os.path.join(models_dir, "hybrid_model.pkl")
vectorizer_path = os.path.join(models_dir, "vectorizer.pkl")
scaler_path = os.path.join(models_dir, "scaler.pkl")

joblib.dump(model, model_path)
joblib.dump(vectorizer, vectorizer_path)
joblib.dump(scaler, scaler_path)

print("\nModel saved at:", model_path)
print("Vectorizer saved at:", vectorizer_path)
print("Scaler saved at:", scaler_path)


# -------------------------------------------------
# Feature Importance (Text Only)
# -------------------------------------------------
feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_[0][:len(feature_names)]

top_positive = sorted(zip(coefficients, feature_names), reverse=True)[:20]
top_negative = sorted(zip(coefficients, feature_names))[:20]

print("\nTop Trending Text Indicators:")
for coef, word in top_positive:
    print(word)

print("\nTop Non-Trending Text Indicators:")
for coef, word in top_negative:
    print(word)
