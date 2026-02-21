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
views_per_video = float(input("Enter channel avg views per video: "))
velocity = float(input("Enter current view velocity (views/hour): "))
duration_sec = float(input("Enter video duration (seconds): "))
publish_hour = int(input("Enter publish hour (0-23): "))


# =========================================================
# Feature Engineering
# =========================================================

like_ratio = likes / views if views > 0 else 0
comment_ratio = comments / views if views > 0 else 0

caps_ratio = (
    sum(1 for c in title if c.isupper()) / len(title)
    if len(title) > 0 else 0
)

title_length = len(title.split())


numeric_data = {
    "title_length": title_length,
    "caps_ratio": caps_ratio,
    "like_ratio": like_ratio,
    "comment_ratio": comment_ratio,
    "velocity": velocity,
    "subscriber_count": subs,
    "views_per_video": views_per_video,
    "duration_sec": duration_sec,
    "publish_hour": publish_hour,
}

numeric_df = pd.DataFrame([numeric_data])

expected_columns = scaler.feature_names_in_
numeric_df = numeric_df.reindex(columns=expected_columns, fill_value=0)

scaled_numeric = scaler.transform(numeric_df)

