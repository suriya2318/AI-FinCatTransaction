import pandas as pd
import os
from glob import glob

CANONICAL_COLS = ["transaction", "merchant", "amount", "label"]


def canonicalize_row(row):
    mapping = {}
    text_candidates = ["transaction", "description", "memo", "notes", "text"]
    merchant_candidates = ["merchant", "vendor", "payee"]
    amount_candidates = ["amount", "amt", "value"]
    label_candidates = ["label", "category", "cat"]

    for c in text_candidates:
        if c in row.index and pd.notna(row[c]):
            mapping["transaction"] = row[c]
            break

    for c in merchant_candidates:
        if c in row.index and pd.notna(row[c]):
            mapping["merchant"] = row[c]
            break

    for c in amount_candidates:
        if c in row.index and pd.notna(row[c]):
            mapping["amount"] = row[c]
            break

    for c in label_candidates:
        if c in row.index and pd.notna(row[c]):
            mapping["label"] = row[c]
            break

    return mapping


def ingest_folder(folder="data/raw", out="data/raw/canonical_transactions.csv"):
    files = glob(os.path.join(folder, "*.csv"))
    rows = []

    for f in files:
        df = pd.read_csv(f, low_memory=False)
        for _, r in df.iterrows():
            mapped = canonicalize_row(r)
            if "transaction" not in mapped:
                mapped["transaction"] = " ".join(
                    str(v)
                    for v in [r.get("merchant", ""), r.get("description", "")]
                    if pd.notna(v)
                )
            rows.append({c: mapped.get(c, None) for c in CANONICAL_COLS})

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    out_df.to_csv(out, index=False)
    print(f"Ingested {len(files)} files -> {out} ({len(out_df)} rows)")
    return out_df


if __name__ == "__main__":
    ingest_folder()
