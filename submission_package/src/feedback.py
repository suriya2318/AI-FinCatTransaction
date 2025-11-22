# src/feedback_cli.py
import csv, os
from src.infer import predict, normalize_text

FEEDBACK_FILE = "data/feedback/feedback.csv"
os.makedirs("data/feedback", exist_ok=True)


def cli():
    while True:
        t = input("Enter transaction (or 'quit')> ")
        if t.strip().lower() in ("q", "quit", "exit"):
            break
        pred, conf = predict([t])[0]
        print(f"Model predicted: {pred} (conf={conf:.3f})")
        correct = input(f"Is this correct? (y/n) ")
        if correct.lower().startswith("y"):
            label = pred
        else:
            label = input(
                "Enter correct label id (e.g. groceries,dining,fuel,shopping,utilities,other)> "
            )
        with open(FEEDBACK_FILE, "a", newline="", encoding="utf8") as f:
            writer = csv.writer(f)
            writer.writerow([t, label])
        print("Saved feedback. Repeat or type quit.")


if __name__ == "__main__":
    cli()
