import csv, os
from src.infer import predict
from src.taxonomy_lookup import get_all_categories

FEEDBACK_FILE = "data/feedback/feedback.csv"
os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)


def cli():
    cats = get_all_categories()
    print("Categories:")
    for i, (cid, cname) in enumerate(cats, start=1):
        print(f"{i}. {cname} ({cid})")
    while True:
        t = input("Enter transaction (or 'quit')> ")
        if t.strip().lower() in ("q", "quit", "exit"):
            break
        out = predict([t])[0]
        print(f"Model predicted: {out['pred']} (conf={out['conf']:.3f})")
        correct = input("Is this correct? (y/n) ")
        if correct.lower().startswith("y"):
            label = out["pred"]
        else:
            sel = input("Enter category number from the list or type label id> ")
            try:
                sel_i = int(sel)
                label = cats[sel_i - 1][0]
            except Exception:
                label = sel.strip()
        with open(FEEDBACK_FILE, "a", newline="", encoding="utf8") as f:
            writer = csv.writer(f)
            writer.writerow([t, label])
        print("Saved feedback.")


if __name__ == "__main__":
    cli()
