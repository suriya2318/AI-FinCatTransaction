# src/app.py (corrected)
import sys
import os

# ensure repo root is on path (your project structure expects imports from src.*)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, render_template, redirect, url_for
import csv

from src.infer import predict
from src.explain import explain_text

# import loader & helpers from taxonomy_lookup
from src.taxonomy_lookup import get_all_categories, alias_lookup, load_taxonomy

app = Flask(__name__)
FEEDBACK_FILE = "data/feedback/feedback.csv"
os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():
    # categories is a list of tuples (id, display_name) used to render the dropdown in the template
    categories = get_all_categories()
    if request.method == "POST":
        text = request.form.get("transaction", "").strip()

        # If saving correction
        if "save" in request.form:
            # Accept multiple possible form fields (dropdown value, free text, legacy names)
            posted_label = (
                request.form.get("label_select")
                or request.form.get("label")
                or request.form.get("label_text")
                or ""
            )
            posted_label = posted_label.strip()

            # Normalize label to canonical category ID before writing
            taxonomy = (
                load_taxonomy()
            )  # returns list of dicts with fields 'id' and 'display_name'
            id_set = {c["id"] for c in taxonomy}
            display_to_id = {
                (c.get("display_name") or "").strip().lower(): c["id"] for c in taxonomy
            }

            label_to_write = posted_label

            if label_to_write == "":
                # nothing provided — write empty string (or you could reject)
                label_to_write = ""
            else:
                # If user posted a numeric index (legacy), map to index in categories if possible
                if label_to_write.isdigit():
                    try:
                        idx = int(label_to_write) - 1
                        # ensure index within range
                        if 0 <= idx < len(categories):
                            label_to_write = categories[idx][0]  # cid
                    except Exception:
                        pass

                # If label already an ID, keep it
                if label_to_write in id_set:
                    pass
                else:
                    # Try to map from display name (case-insensitive)
                    lower = label_to_write.lower()
                    if lower in display_to_id:
                        label_to_write = display_to_id[lower]
                    else:
                        # As a last resort: try to match by partial token to id keys (conservative)
                        matched = None
                        for c in taxonomy:
                            if label_to_write.lower() == (c.get("id") or "").lower():
                                matched = c["id"]
                                break
                        if matched:
                            label_to_write = matched
                        else:
                            # keep as provided (this will create a custom/new label id)
                            # Optionally you can reject this and ask user to pick from dropdown.
                            label_to_write = label_to_write

            # persist transaction text and canonical label id (not display name)
            with open(FEEDBACK_FILE, "a", newline="", encoding="utf8") as f:
                writer = csv.writer(f)
                writer.writerow([text, label_to_write])

            return redirect(url_for("index"))

        # Prediction flow
        out = predict([text], top_k=5)[0]
        pred = out["pred"]
        conf = out["conf"]
        candidates = out.get("candidates")
        alias_override = out.get("alias_override")

        # Explanations from your existing explain module (should return list of (feat, score) or (pred, conf, list))
        expl = []
        try:
            maybe = explain_text(text)
            # explain_text might return:
            # 1) a list of (feat,score)
            # 2) a tuple (pred, conf, [(feat,score), ...])
            if isinstance(maybe, list):
                expl = maybe
            elif isinstance(maybe, tuple) and len(maybe) >= 3:
                expl = maybe[2] if isinstance(maybe[2], list) else []
            else:
                # fallback: try to coerce to list
                expl = list(maybe) if maybe else []
        except Exception:
            expl = []

        # normalize explanation entries into list of (feature, float_score)
        norm_expl = []
        for item in expl or []:
            try:
                feat = str(item[0])
                score = float(item[1])
            except Exception:
                feat = str(item[0]) if item else "feature"
                score = 0.0
            norm_expl.append((feat, score))

        # prepare chart labels if template uses them
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
            candidates=candidates,
            alias_override=alias_override,
        )

    # GET
    return render_template(
        "index.html", categories=categories, chart_labels=[], chart_scores=[]
    )


if __name__ == "__main__":
    app.run(port=8787, debug=True)
