import os
import joblib
import pandas as pd
import numpy as np

# =========================================================
# PATH SETUP (PORTABLE)
# =========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "hybrid_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "vectorizer.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

# =========================================================
# LOAD MODEL FILES
# =========================================================
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
scaler = joblib.load(SCALER_PATH)

print("\nTrendPulse Prediction System")
print("=" * 40)

# =========================================================
# USER INPUT
# =========================================================
title = input("Enter video title: ")
views = float(input("Enter current views: "))
likes = float(input("Enter current likes: "))
comments = float(input("Enter current comments: "))
subs = float(input("Enter channel subscriber count: "))
duration_sec = float(input("Enter video duration (seconds): "))
publish_hour = int(input("Enter publish hour (0-23): "))

# =========================================================
# FEATURE ENGINEERING (MATCH TRAINING FEATURES)
# =========================================================

# Safe division
like_ratio = likes / views if views > 0 else 0
comment_ratio = comments / views if views > 0 else 0

# Title features
caps_ratio = (
    sum(1 for c in title if c.isupper()) / len(title)
    if len(title) > 0 else 0
)

title_length = len(title.split())

# =========================================================
# BUILD NUMERIC DATAFRAME
# IMPORTANT: Must match training feature names exactly
# =========================================================

numeric_data = {
    "like_ratio": like_ratio,
    "comment_ratio": comment_ratio,
    "caps_ratio": caps_ratio,
    "duration_sec": duration_sec,
    "publish_hour": publish_hour,
    "title_length": title_length,
}

numeric_df = pd.DataFrame([numeric_data])

# Ensure correct column order
expected_columns = scaler.feature_names_in_
numeric_df = numeric_df.reindex(columns=expected_columns, fill_value=0)

# Scale numeric features
scaled_numeric = scaler.transform(numeric_df)

# =========================================================
# TEXT FEATURES
# =========================================================
text_features = vectorizer.transform([title])

# Combine text + numeric
X_final = np.hstack((text_features.toarray(), scaled_numeric))

# =========================================================
# PREDICTION
# =========================================================
prediction = model.predict(X_final)[0]
probabilities = model.predict_proba(X_final)[0]

prob_non_viral = probabilities[0]
prob_viral = probabilities[1]

# =========================================================
# SMART THRESHOLD
# =========================================================
if prob_viral > 0.75:
    confidence = "🔥 HIGH Viral Potential"
elif prob_viral > 0.60:
    confidence = "⚡ Moderate Viral Potential"
else:
    confidence = "❄ Low Viral Potential"

# =========================================================
# OUTPUT
# =========================================================
print("\nPrediction Results")
print("=" * 40)
print(f"Viral Probability: {prob_viral:.2f}")
print(f"Non-Viral Probability: {prob_non_viral:.2f}")
print(f"Model Decision: {'VIRAL' if prediction == 1 else 'NON-VIRAL'}")
print(f"Confidence Level: {confidence}")