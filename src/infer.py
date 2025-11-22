# src/infer.py
import joblib
import numpy as np
from src.preprocess import normalize_text
from src.taxonomy_lookup import alias_lookup
import os

MODEL_PATH = os.path.join("artifacts", "checkpoints", "baseline.joblib")
_model = None


def _load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def predict(texts, top_k=3):
    """
    Return list of dicts for each input text:
      {
        'text': original_text,
        'pred': top_pred_label,
        'conf': top_confidence (float),
        'candidates': [(label, prob), ...],
        'alias_override': (label or None),
      }
    """
    model = _load_model()
    cleaned = [normalize_text(t) for t in texts]
    probs = model.predict_proba(cleaned)
    classes = model.classes_
    results = []
    for i, t in enumerate(texts):
        prob_row = probs[i]
        idxs = np.argsort(-prob_row)
        candidates = [(classes[idx], float(prob_row[idx])) for idx in idxs[:top_k]]
        top_label, top_prob = candidates[0]
        alias_cat, method = alias_lookup(t)
        alias_override = None
        if alias_cat and method == "token":
            alias_override = alias_cat
            top_label = alias_cat
            top_prob = max(top_prob, 0.99)
        results.append(
            {
                "text": t,
                "pred": top_label,
                "conf": float(top_prob),
                "candidates": candidates,
                "alias_override": alias_override,
            }
        )
    return results


if __name__ == "__main__":
    examples = ["Starbucks 2345", "AMAZON MKTPLACE", "SHELL GAS STATION", "shall shop"]
    out = predict(examples, top_k=3)
    for o in out:
        print(o)
