import joblib
import numpy as np
from typing import List, Dict, Any

from src.preprocess import normalize_text
from src.taxonomy_lookup import alias_lookup

MODEL_PATH = "artifacts/checkpoints/baseline.joblib"

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model from '{MODEL_PATH}': {e}")


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over last axis."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    x = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(x)
    return exp / exp.sum(axis=1, keepdims=True)


def _to_candidates(
    classes: List[str], probs_row: np.ndarray, top_k: int = 5
) -> List[Dict[str, Any]]:
    """Return list of candidate dicts sorted by probability desc."""
    idx = np.argsort(probs_row)[::-1]
    candidates = []
    for i in idx[:top_k]:
        candidates.append({"id": str(classes[i]), "prob": float(probs_row[i])})
    return candidates


def predict(texts: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Predict categories for a list of transaction texts.

    Returns a list of result dicts for each input text:
      {
        "pred": "<category_id>",
        "conf": 0.9123,              # top probability/confidence
        "candidates": [{"id": "...", "prob": 0.9}, ...],
        "alias_override": True|False
      }

    - Handles models with predict_proba or decision_function.
    - If an alias match is found (token alias), it returns an alias override with high confidence.
    """
    if not isinstance(texts, (list, tuple)):
        texts = [texts]

    cleaned = [normalize_text(t) for t in texts]

    results = []

    for orig_text, ctext in zip(texts, cleaned):
        try:
            alias_cat, method = alias_lookup(orig_text)
        except Exception:
            alias_cat, method = (None, None)

        if alias_cat and method == "token":
            results.append(
                {
                    "pred": alias_cat,
                    "conf": 0.99,
                    "candidates": [{"id": alias_cat, "prob": 0.99}],
                    "alias_override": True,
                }
            )
            continue
        try:
            probs = model.predict_proba([ctext])
            classes = getattr(model, "classes_", None)
            if classes is None:
                try:
                    last = (
                        getattr(model, "steps", [])[-1][1]
                        if getattr(model, "steps", None)
                        else None
                    )
                    classes = getattr(last, "classes_", None)
                except Exception:
                    classes = None
            if classes is None:
                raise RuntimeError(
                    "Model has no classes_ attribute after predict_proba"
                )
            probs_row = probs[0]
        except Exception:
            try:
                scores = model.decision_function([ctext])
                scores_arr = np.asarray(scores)
                if scores_arr.ndim == 1:
                    scores_arr = scores_arr.reshape(1, -1)
                probs_row = _softmax(scores_arr)[0]
                classes = getattr(model, "classes_", None)
                if classes is None:
                    try:
                        last = (
                            getattr(model, "steps", [])[-1][1]
                            if getattr(model, "steps", None)
                            else None
                        )
                        classes = getattr(last, "classes_", None)
                    except Exception:
                        classes = None
                if classes is None:
                    classes = [str(i) for i in range(probs_row.shape[0])]
            except Exception as e:
                classes = getattr(model, "classes_", None)
                if classes is None:
                    classes = ["other"]
                results.append(
                    {
                        "pred": str(classes[0]),
                        "conf": 0.0,
                        "candidates": [{"id": str(classes[0]), "prob": 0.0}],
                        "alias_override": False,
                    }
                )
                continue
        candidates = _to_candidates(classes, probs_row, top_k=top_k)
        top = candidates[0]
        results.append(
            {
                "pred": top["id"],
                "conf": float(top["prob"]),
                "candidates": candidates,
                "alias_override": False,
            }
        )

    return results


def predict_single(text: str, top_k: int = 5) -> Dict[str, Any]:
    return predict([text], top_k=top_k)[0]


if __name__ == "__main__":
    # quick smoke test examples (only run locally)
    examples = ["Starbucks 2345", "AMAZON MKTPLACE", "SHELL GAS STATION", "Coffee"]
    out = predict(examples, top_k=3)
    for t, o in zip(examples, out):
        print(
            t,
            "->",
            o["pred"],
            f"(conf={o['conf']:.3f})",
            "candidates:",
            o["candidates"],
        )
