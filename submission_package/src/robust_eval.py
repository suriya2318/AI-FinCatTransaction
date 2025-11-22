import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import GroupShuffleSplit


def merchant_proxy(text):
    if not isinstance(text, str) or text.strip() == "":
        return "UNK"
    return text.split()[0].lower()


def run_merchant_split_eval(
    processed_csv="data/processed/processed.csv", random_state=42
):
    # Ensure output dirs
    os.makedirs("artifacts/metrics", exist_ok=True)
    os.makedirs("artifacts/checkpoints", exist_ok=True)

    # Load data
    df = pd.read_csv(processed_csv)

    # Ensure required columns
    if "text" not in df.columns:
        if "transaction" in df.columns:
            df = df.rename(columns={"transaction": "text"})
        else:
            raise ValueError("Missing 'text' or 'transaction' column in input CSV.")
    if "label" not in df.columns:
        raise ValueError("Missing 'label' column in input CSV.")

    # Group by merchant proxy (first token)
    df["merchant_proxy"] = df["text"].apply(merchant_proxy)
    print("Unique merchant proxies:", df["merchant_proxy"].nunique())

    # Group-based split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    train_idx, test_idx = next(gss.split(df, groups=df["merchant_proxy"]))
    train = df.iloc[train_idx]
    test = df.iloc[test_idx]

    X_train = train["text"].astype(str)
    y_train = train["label"].astype(str)
    X_test = test["text"].astype(str)
    y_test = test["label"].astype(str)

    # Model pipeline
    pipe = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), analyzer="char_wb", max_features=5000),
        LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=random_state
        ),
    )
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    # Classification report (console + JSON)
    print("=== Merchant-group split evaluation ===")
    report_dict = classification_report(
        y_test, y_pred, output_dict=True, zero_division=0
    )
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    with open(
        "artifacts/metrics/classification_report.json", "w", encoding="utf-8"
    ) as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False)
    print(
        "✅ Saved classification report to artifacts/metrics/classification_report.json"
    )

    # Confusion matrix (only for labels present in test)
    labels_in_test = [label for label in pipe.classes_ if label in set(y_test)]
    if labels_in_test:
        cm = confusion_matrix(y_test, y_pred, labels=labels_in_test)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=labels_in_test
        )
        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        out_path = "artifacts/metrics/confusion_matrix.png"
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"✅ Saved confusion matrix to {out_path}")
    else:
        print("⚠️ No labels in test set match model classes. Skipping confusion matrix.")

    # Save model
    joblib.dump(pipe, "artifacts/checkpoints/merchant_split.joblib")
    print("✅ Saved model to artifacts/checkpoints/merchant_split.joblib")


if __name__ == "__main__":
    run_merchant_split_eval()
