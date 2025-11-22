# src/crossval_eval.py
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np

df = pd.read_csv("data/processed/processed.csv")
if "text" not in df.columns:
    df = (
        df.rename(columns={"transaction": "text"})
        if "transaction" in df.columns
        else df
    )
X = df["text"]
y = df["label"]
pipe = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), analyzer="char_wb", max_features=5000),
    LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
print("5-fold macro F1 scores:", scores)
print("Mean:", np.mean(scores), "Std:", np.std(scores))
