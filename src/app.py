# src/app.py
import sys
import os

# allow imports like `from src.infer import predict`
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, render_template, redirect, url_for
import csv

from src.infer import predict
from src.explain import explain_text
from src.taxonomy_lookup import get_all_categories, alias_lookup, load_taxonomy

app = Flask(__name__)

FEEDBACK_FILE = "data/feedback/feedback.csv"
os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)


def get_categories():
    """
    Returns list of tuples (category_id, display_name) for the UI.
    """
    return get_all_categories()


@app.route("/", methods=["GET", "POST"])
def index():
    categories = get_categories()
    alias_override = None

    if request.method == "POST":
        text = request.form.get("transaction", "").strip()

        # --- Save feedback flow ---
        if "save" in request.form:
            # prefer dropdown value (category id) else free-text label
            posted_label = (
                request.form.get("label_select")
                or request.form.get("label")
                or request.form.get("label_text")
                or ""
            )
            posted_label = posted_label.strip()

            # Map posted label/display name to canonical id if possible
            taxonomy = load_taxonomy()
            id_set = {c["id"] for c in taxonomy}
            display_to_id = {
                (c.get("display_name") or "").strip().lower(): c["id"] for c in taxonomy
            }

            label_to_write = posted_label

            # If numeric legacy index provided, map to categories list
            if label_to_write.isdigit():
                try:
                    idx = int(label_to_write) - 1
                    if 0 <= idx < len(categories):
                        label_to_write = categories[idx][0]
                except Exception:
                    pass

            # If user submitted display name, map to id
            if label_to_write and label_to_write not in id_set:
                low = label_to_write.lower()
                if low in display_to_id:
                    label_to_write = display_to_id[low]

            # Persist transaction text and canonical label id (or free text if unmatched)
            with open(FEEDBACK_FILE, "a", newline="", encoding="utf8") as f:
                writer = csv.writer(f)
                writer.writerow([text, label_to_write])

            # After saving return to entry (GET index) so user can add next transaction
            return redirect(url_for("index"))

        # --- Prediction flow ---
        if not text:
            # nothing to predict — just reload
            return redirect(url_for("index"))

        out = predict([text], top_k=5)[0]
        pred = out.get("pred")
        conf = out.get("conf", 0.0)
        candidates = out.get("candidates", [])
        alias_override = out.get("alias_override")

        # Explainability: expect explain_text to return list[(feat, score)] or (pred, conf, expl_list)
        expl = []
        try:
            maybe = explain_text(text)
            if isinstance(maybe, list):
                expl = maybe
            elif (
                isinstance(maybe, tuple)
                and len(maybe) >= 3
                and isinstance(maybe[2], list)
            ):
                expl = maybe[2]
            else:
                # try to coerce if possible
                if hasattr(maybe, "__iter__"):
                    expl = list(maybe)
        except Exception:
            expl = []

        # normalize explanation entries into (feature, float_score)
        norm_expl = []
        for item in expl or []:
            try:
                feat = str(item[0])
                score = float(item[1])
            except Exception:
                feat = str(item[0]) if item else "feature"
                score = 0.0
            norm_expl.append((feat, score))

        chart_labels = [f for f, s in norm_expl]
        chart_scores = [s for f, s in norm_expl]

        # Render results template (index.html expects these variables)
        return render_template(
            "index.html",
            text=text,
            pred=pred,
            conf=conf,
            expl=norm_expl,
            categories=categories,
            chart_labels=chart_labels,
            chart_scores=chart_scores,
            candidates=candidates,
            alias_override=alias_override,
        )

    # GET request: show entry form
    return render_template(
        "index.html", categories=categories, chart_labels=[], chart_scores=[]
    )


if __name__ == "__main__":
    app.run(port=8787, debug=True)
