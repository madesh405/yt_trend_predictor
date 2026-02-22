import os
import joblib
import numpy as np
import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# -------------------------------------------------
# Load Model + Scaler
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(BASE_DIR, "models")

model = joblib.load(os.path.join(models_dir, "early_model.pkl"))
scaler = joblib.load(os.path.join(models_dir, "early_scaler.pkl"))

analyzer = SentimentIntensityAnalyzer()


# -------------------------------------------------
# NLP Feature Extraction (MUST MATCH TRAINING)
# -------------------------------------------------
def extract_nlp_features(text):
    blob = TextBlob(text)

    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    vader_score = analyzer.polarity_scores(text)["compound"]

    exclamations = text.count("!")
    questions = text.count("?")

    words = text.split()
    word_count = len(words)
    unique_words = len(set(words))
    lexical_diversity = unique_words / (word_count + 1)

    return {
        "polarity": polarity,
        "subjectivity": subjectivity,
        "vader_score": vader_score,
        "exclamations": exclamations,
        "questions": questions,
        "lexical_diversity": lexical_diversity
    }


# -------------------------------------------------
# User Input
# -------------------------------------------------
print("\nTrendPulse Early Prediction System")
print("========================================")

title = input("Enter video title: ")
description = input("Enter description (optional): ")

subscriber_count = float(input("Subscriber count: "))
views_per_video = float(input("Channel avg views per video: "))
duration_sec = float(input("Video duration (seconds): "))
publish_hour = float(input("Publish hour (0-23): "))
likes = float(input("Current likes: "))
comments = float(input("Current comments: "))
views = float(input("Current views: "))
age_hours = float(input("Hours since upload: "))


# -------------------------------------------------
# Feature Engineering
# -------------------------------------------------
full_text = title + " " + description

title_length = len(title)
caps_ratio = sum(1 for c in title if c.isupper()) / (len(title) + 1)

like_ratio = likes / (views + 1)
comment_ratio = comments / (views + 1)

velocity = views / (age_hours + 1)


# NLP Features
nlp_features = extract_nlp_features(full_text)


# -------------------------------------------------
# Build Feature DataFrame (SAFE VERSION)
# -------------------------------------------------
feature_dict = {
    "title_length": title_length,
    "caps_ratio": caps_ratio,
    "duration_sec": duration_sec,
    "publish_hour": publish_hour,
    "subscriber_count": subscriber_count,
    "views_per_video": views_per_video,
    "like_ratio": like_ratio,
    "comment_ratio": comment_ratio,
    "velocity": velocity,
    "polarity": nlp_features["polarity"],
    "subjectivity": nlp_features["subjectivity"],
    "vader_score": nlp_features["vader_score"],
    "exclamations": nlp_features["exclamations"],
    "questions": nlp_features["questions"],
    "lexical_diversity": nlp_features["lexical_diversity"]
}

feature_df = pd.DataFrame([feature_dict])


# -------------------------------------------------
# Scale + Predict
# -------------------------------------------------
feature_scaled = scaler.transform(feature_df)

probability = model.predict_proba(feature_scaled)[0][1]
prediction = 1 if probability >= 0.5 else 0


# -------------------------------------------------
# Confidence Logic
# -------------------------------------------------
if probability >= 0.80:
    confidence = "VERY HIGH"
elif probability >= 0.65:
    confidence = "HIGH"
elif probability >= 0.55:
    confidence = "MODERATE"
else:
    confidence = "LOW"


# -------------------------------------------------
# Output
# -------------------------------------------------
print("\n==============================")
print("Prediction Results")
print("==============================")

print(f"Viral Probability: {min(probability, 0.999):.4f}")

if prediction == 1:
    print("Prediction: 🔥 LIKELY VIRAL (Relative to Channel)")
else:
    print("Prediction: ❄️ Likely Non-Viral")

print("Confidence:", confidence)
