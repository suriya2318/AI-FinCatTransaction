import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
from src.preprocess import load_and_process


def train(
    path="data/processed/processed.csv",
    model_out="artifacts/checkpoints/baseline.joblib",
):
    df = load_and_process()
    X = df["text"]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipe = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), max_features=5000, analyzer="char_wb"),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    )
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print(classification_report(y_test, y_pred, digits=4))
    cm = confusion_matrix(y_test, y_pred, labels=pipe.classes_)
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump(pipe, model_out)
    print("Saved model to", model_out)
    return pipe, X_test, y_test


if __name__ == "__main__":
    train()
