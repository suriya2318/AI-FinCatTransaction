import pandas as pd
import os
from src.preprocess import load_and_process, normalize_text
from src.train_baseline import train

FEEDBACK_FILE = "data/feedback/feedback.csv"
PROCESSED_FILE = "data/processed/processed.csv"
MERGED_FILE = "data/processed/merged_for_retrain.csv"


def merge_and_retrain():
    if not os.path.exists(PROCESSED_FILE):
        print("Processed file not found. Running preprocessing...")
        load_and_process()

    base = pd.read_csv(PROCESSED_FILE)
    if "text" in base.columns:
        base_df = base[["text", "label"]].dropna(subset=["label"])
    elif "transaction" in base.columns:
        base_df = (
            base[["transaction", "label"]]
            .rename(columns={"transaction": "text"})
            .dropna(subset=["label"])
        )
    else:
        raise SystemExit(
            "Processed file must contain 'text' or 'transaction' and 'label'."
        )

    if not os.path.exists(FEEDBACK_FILE):
        print("No feedback file found. Nothing to merge.")
        return

    fb = pd.read_csv(FEEDBACK_FILE, header=None, names=["transaction", "label"])
    fb["text"] = fb["transaction"].astype(str).apply(normalize_text)
    fb = fb[["text", "label"]]

    merged = pd.concat([base_df, fb], ignore_index=True)
    os.makedirs(os.path.dirname(MERGED_FILE), exist_ok=True)
    merged.to_csv(MERGED_FILE, index=False)
    print(f"Merged dataset saved to {MERGED_FILE} (rows={len(merged)})")
    train(input_csv=MERGED_FILE)
    print("Retraining complete. Model updated.")


if __name__ == "__main__":
    merge_and_retrain()
