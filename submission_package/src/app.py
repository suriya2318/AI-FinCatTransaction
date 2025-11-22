# src/app.py
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, render_template, redirect, url_for
import csv
from src.infer import predict
from src.explain import explain_text
from src.taxonomy_lookup import load_taxonomy, alias_lookup

app = Flask(__name__)
FEEDBACK_FILE = "data/feedback/feedback.csv"
os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)


# Helper to get list of category ids and display names
def get_categories():
    cats = load_taxonomy()
    # return list of tuples (id, display_name)
    return [(c["id"], c.get("display_name", c["id"])) for c in cats]


@app.route("/", methods=["GET", "POST"])
def index():
    categories = get_categories()
    if request.method == "POST":
        text = request.form.get("transaction", "").strip()
        # Save feedback if present via dropdown
        if "save" in request.form:
            label = request.form.get("label_select") or request.form.get("label") or ""
            with open(FEEDBACK_FILE, "a", newline="", encoding="utf8") as f:
                writer = csv.writer(f)
                writer.writerow([text, label])
            return redirect(url_for("index"))

        # Prediction flow
        alias_cat, method = alias_lookup(text)
        if alias_cat and method == "token":
            pred = alias_cat
            conf = 0.99
            expl = [("alias_match", 0.99)]
        else:
            pred, conf, expl = explain_text(text)

        # Normalize explanation list into feature/score floats
        norm_expl = []
        for item in expl or []:
            try:
                feat = str(item[0])
                score = float(item[1])
            except Exception:
                feat = str(item[0]) if item else "feature"
                score = 0.0
            norm_expl.append((feat, score))

        # For chart data: labels and scores
        chart_labels = [f for f, s in norm_expl]
        chart_scores = [s for f, s in norm_expl]

        return render_template(
            "index.html",
            text=text,
            pred=pred,
            conf=conf,
            expl=norm_expl,
            categories=categories,
            chart_labels=chart_labels,
            chart_scores=chart_scores,
        )

    # GET
    return render_template(
        "index.html", categories=categories, chart_labels=[], chart_scores=[]
    )


if __name__ == "__main__":
    app.run(port=8787, debug=True)
