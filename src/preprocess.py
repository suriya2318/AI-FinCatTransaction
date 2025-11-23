import pandas as pd
import re
import unicodedata


def normalize_text(s):
    if pd.isna(s):
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s)
    s = s.lower()
    s = re.sub(r"\d{4,}", " <NUM> ", s)
    s = re.sub(r"[\W_]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_and_process(
    path_in="data/raw/transaction_synthetic.csv",
    path_out="data/processed/processed.csv",
):
    df = pd.read_csv(path_in)
    df["text"] = df["transaction"].apply(normalize_text)
    df[["text", "label"]].to_csv(path_out, index=False)
    print("Wrote", path_out)
    return df


if __name__ == "__main__":
    load_and_process()
