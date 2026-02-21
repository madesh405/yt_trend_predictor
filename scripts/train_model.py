import os
import pandas as pd
import joblib
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_auc_score,
    precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix

from xgboost import XGBClassifier


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
# Feature Definitions
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
groups = df["channel_id"]


# -------------------------------------------------
# TF-IDF
# -------------------------------------------------
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=8000,
    ngram_range=(1, 2),
    min_df=2
)

X_text_vec = vectorizer.fit_transform(X_text)


# -------------------------------------------------
# Scale Numeric
# -------------------------------------------------
scaler = StandardScaler()
X_numeric_scaled = scaler.fit_transform(X_numeric)
X_numeric_sparse = csr_matrix(X_numeric_scaled)


# -------------------------------------------------
# Combine Features
# -------------------------------------------------
X_final = hstack([X_text_vec, X_numeric_sparse])


# -------------------------------------------------
# Channel-wise Split
# -------------------------------------------------
gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
train_idx, test_idx = next(gss.split(X_final, y, groups))

X_train, X_test = X_final[train_idx], X_final[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]


# -------------------------------------------------
# XGBoost Model
# -------------------------------------------------
model = XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    use_label_encoder=False
)

model.fit(X_train, y_train)


# -------------------------------------------------
# Evaluate
# -------------------------------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)

print("\n=== DEFAULT THRESHOLD (0.5) ===")
print("Test Accuracy:", accuracy)
print("ROC AUC:", roc)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# -------------------------------------------------
# Optimal Threshold (F1)
# -------------------------------------------------
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)

f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

best_index = np.argmax(f1_scores)
best_threshold = thresholds[best_index]

print("\nBest Threshold (F1 optimized):", best_threshold)
print("Best F1 Score:", f1_scores[best_index])

y_pred_optimal = (y_prob >= best_threshold).astype(int)

print("\n=== OPTIMIZED THRESHOLD RESULTS ===")
print(classification_report(y_test, y_pred_optimal))


# -------------------------------------------------
# Save Everything
# -------------------------------------------------
joblib.dump(model, os.path.join(models_dir, "hybrid_model.pkl"))
joblib.dump(vectorizer, os.path.join(models_dir, "vectorizer.pkl"))
joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))

print("\nXGBoost Model Saved Successfully.")