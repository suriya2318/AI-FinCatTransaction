# src/infer.py
import joblib, pandas as pd
from src.preprocess import normalize_text
import numpy as np

model = joblib.load("artifacts/checkpoints/baseline.joblib")
def predict(texts, top_k=1):
    cleaned = [normalize_text(t) for t in texts]
    probs = model.predict_proba(cleaned)
    preds = model.classes_[probs.argmax(axis=1)]
    confidences = probs.max(axis=1)
    return list(zip(preds, confidences))

if __name__=="__main__":
    examples = ["Starbucks 2345", "AMAZON MKTPLACE", "SHELL GAS STATION"]
    for p,c in predict(examples):
        print(examples, "=>", p, round(c,3))
