import os
import re
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load datasets

fake_df = pd.read_csv("dataset/Fake.csv")
true_df = pd.read_csv("dataset/True.csv")

print("Fake dataset:", fake_df.shape)
print("Real dataset:", true_df.shape)

# 2. Assign labels

fake_df["label"] = 0
true_df["label"] = 1

# 3. Combine datasets

df = pd.concat([fake_df, true_df], ignore_index=True)

# 4. Prepare text

# Combine title and article body if available
df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")

# Remove empty records
df = df[df["text"].str.strip() != ""]

# Remove duplicate articles
df.drop_duplicates(subset=["text"], inplace=True)

# 5. Shuffle dataset

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nCombined dataset:", df.shape)
print("\nLabel distribution:")
print(df["label"].value_counts())

# 6. Separate features and labels

X = df["text"]
y = df["label"]

# 7. Split dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# 8. TF-IDF Vectorization

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# 9. Train Logistic Regression

model = LogisticRegression(
    max_iter=1000,
    C=1.0,
    random_state=42
)

print("\nTraining model...")

model.fit(X_train_vectorized, y_train)

# 10. Predictions

y_pred = model.predict(X_test_vectorized)

# 11. Evaluation

accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL RESULTS ==========")

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Fake News", "Real News"]
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 12. Save model and vectorizer

os.makedirs("models", exist_ok=True)

joblib.dump(vectorizer, "models/vectorizer.jb")
joblib.dump(model, "models/lr_model.jb")

print("\nModel saved successfully!")

