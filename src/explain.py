# src/explain.py (top)
import joblib
import numpy as np
from src.preprocess import normalize_text

# Optional SHAP import — do not crash if shap is missing
try:
    import shap

    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False

# load model later (same as before)
MODEL_PATH = "artifacts/checkpoints/baseline.joblib"
_model = None
_vectorizer = None
_clf = None


def _load_model():
    global _model, _vectorizer, _clf
    if _model is None:
        _model = joblib.load(MODEL_PATH)
        _vectorizer = _model.named_steps["tfidfvectorizer"]
        _clf = _model.named_steps["logisticregression"]


def explain_text(text, nsamples=100):
    _load_model()
    txt = normalize_text(text)
    Xv = _vectorizer.transform([txt])
    probs = _clf.predict_proba(Xv)[0]
    pred = _clf.classes_[int(np.argmax(probs))]
    conf = float(max(probs))
    explanations = []

    if _HAS_SHAP:
        try:
            # build small explainer (KernelExplainer can be slow)
            background = _vectorizer.transform(
                ["amazon walmart shell starbucks whole foods"]
            )
            explainer = shap.KernelExplainer(_clf.predict_proba, background)
            shap_values = explainer.shap_values(Xv, nsamples=nsamples)
            class_idx = int(np.argmax(probs))
            coefs = shap_values[class_idx][0]
            feat_names = _vectorizer.get_feature_names_out()
            top_idx = np.argsort(-np.abs(coefs))[:8]
            explanations = [(feat_names[i], float(coefs[i])) for i in top_idx]
        except Exception:
            # fallback to linear coef contribution
            pass

    if not explanations:
        # fallback: contribution = coef * feature_value for linear models
        if hasattr(_clf, "coef_"):
            Xarr = Xv.toarray()[0]
            feat_names = _vectorizer.get_feature_names_out()
            class_idx = int(np.argmax(probs))
            contribs = _clf.coef_[class_idx] * Xarr
            top_idx = np.argsort(-np.abs(contribs))[:8]
            explanations = [(feat_names[i], float(contribs[i])) for i in top_idx]

    return pred, conf, explanations
