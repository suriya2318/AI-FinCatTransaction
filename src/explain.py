import joblib
import os
import numpy as np
from src.preprocess import normalize_text

MODEL_PATH = os.path.join("artifacts", "checkpoints", "baseline.joblib")
_model = None
_vectorizer = None
_clf = None


def _load():
    global _model, _vectorizer, _clf
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
        try:
            _vectorizer = _model.named_steps.get(
                "tfidfvectorizer"
            ) or _model.named_steps.get("tfidfvectorizer")
            _clf = (
                _model.named_steps.get("logisticregression")
                or _model.named_steps.get("svc")
                or _model.named_steps.get("linearSVC")
                or _model.named_steps.get("classifier")
            )
        except Exception:
            _vectorizer = None
            _clf = None


def explain_text(text, top_n=8):
    """
    Returns list of (feature, score) sorted desc by absolute contribution.
    If SHAP explainer is not available we use linear coefficient approximation.
    """
    _load()
    txt = normalize_text(text)
    if _vectorizer is not None and hasattr(_clf, "coef_"):
        Xv = _vectorizer.transform([txt])
        arr = Xv.toarray()[0]
        if hasattr(_clf, "predict_proba"):
            probs = _clf.predict_proba(Xv)[0]
            class_idx = int(np.argmax(probs))
        else:
            class_idx = 0
        contribs = _clf.coef_[class_idx] * arr
        feat_names = _vectorizer.get_feature_names_out()
        idxs = np.argsort(-np.abs(contribs))[:top_n]
        res = []
        for i in idxs:
            if arr[i] == 0 and abs(contribs[i]) < 1e-9:
                continue
            res.append((feat_names[i], float(contribs[i])))
        return res
    words = txt.split()
    feats = [
        ("length", float(len(txt))),
        ("word_count", float(len(words))),
    ]
    return feats
