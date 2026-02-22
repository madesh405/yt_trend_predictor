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
# NUMERIC FEATURES ONLY (Signal Test)
# -------------------------------------------------
NUMERIC_COLUMNS = [
    "title_length",
    "caps_ratio",
    "duration_sec",
    "publish_hour",
    "subscriber_count",
    "views_per_video"
]

X = df[NUMERIC_COLUMNS].fillna(0)
y = df["viral"]
groups = df["channel_id"]


# -------------------------------------------------
# Scale Features
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
# 1️⃣ Logistic Regression (Signal Test Model)
# =================================================
print("\n==============================")
print("LOGISTIC REGRESSION RESULTS")
print("==============================")

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)
y_prob_lr = lr_model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("ROC AUC:", roc_auc_score(y_test, y_prob_lr))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr))


# =================================================
# 2️⃣ XGBoost (Only If Signal Exists)
# =================================================
print("\n==============================")
print("XGBOOST RESULTS")
print("==============================")

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("ROC AUC:", roc_auc_score(y_test, y_prob_xgb))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_xgb))


# -------------------------------------------------
# Save Best Model (Based on ROC AUC)
# -------------------------------------------------
roc_lr = roc_auc_score(y_test, y_prob_lr)
roc_xgb = roc_auc_score(y_test, y_prob_xgb)

if roc_lr >= roc_xgb:
    print("\nSaving Logistic Regression Model")
    joblib.dump(lr_model, os.path.join(models_dir, "early_model.pkl"))
else:
    print("\nSaving XGBoost Model")
    joblib.dump(xgb_model, os.path.join(models_dir, "early_model.pkl"))

joblib.dump(scaler, os.path.join(models_dir, "early_scaler.pkl"))

print("\nModel Saved Successfully.")