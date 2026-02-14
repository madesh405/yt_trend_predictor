import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


# -------------------------------------------------
# Paths
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "trendpulse_dataset.csv")
models_dir = os.path.join(BASE_DIR, "models")

os.makedirs(models_dir, exist_ok=True)


# -------------------------------------------------
# Load Dataset
# -------------------------------------------------
df = pd.read_csv(data_path)

# Shuffle dataset for safety
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nDataset Size:", len(df))
print("\nClass Distribution:")
print(df["viral"].value_counts())


X = df["title"]
y = df["viral"]


# -------------------------------------------------
# TF-IDF Vectorization (Upgraded)
# -------------------------------------------------
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=7000,
    ngram_range=(1, 2),   # unigrams + bigrams
    min_df=2
)

X_vec = vectorizer.fit_transform(X)


# -------------------------------------------------
# Train/Test Split
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_vec,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -------------------------------------------------
# Train Model (Balanced Logistic Regression)
# -------------------------------------------------
model = LogisticRegression(
    max_iter=3000,
    class_weight="balanced",
    C=1.5
)

model.fit(X_train, y_train)


# -------------------------------------------------
# Evaluate on Test Set
# -------------------------------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nTest Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# -------------------------------------------------
# Cross-Validation (More Reliable Metric)
# -------------------------------------------------
cv_scores = cross_val_score(model, X_vec, y, cv=5)

print("\nCross Validation Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())


# -------------------------------------------------
# Save Model
# -------------------------------------------------
model_path = os.path.join(models_dir, "title_model.pkl")
vectorizer_path = os.path.join(models_dir, "vectorizer.pkl")

joblib.dump(model, model_path)
joblib.dump(vectorizer, vectorizer_path)

print("\nModel saved at:", model_path)
print("Vectorizer saved at:", vectorizer_path)

feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_[0]

top_positive = sorted(zip(coefficients, feature_names), reverse=True)[:20]
top_negative = sorted(zip(coefficients, feature_names))[:20]

print("\nTop Trending Indicators:")
for coef, word in top_positive:
    print(word)

print("\nTop Non-Trending Indicators:")
for coef, word in top_negative:
    print(word)
