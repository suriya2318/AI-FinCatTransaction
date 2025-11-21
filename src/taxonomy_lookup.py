import yaml
import re
import pandas as pd
from functools import lru_cache


@lru_cache()
def load_taxonomy(path="configs/taxonomy.yaml"):
    with open(path, "r") as f:
        tax = yaml.safe_load(f)

    # Normalize aliases
    categories = tax.get("categories", [])
    for c in categories:
        c["aliases_norm"] = [
            a.lower() for a in c.get("aliases", []) if isinstance(a, str)
        ]
        c["display_name"] = c.get("display_name", c.get("id"))
    return categories


def alias_lookup(text):
    text = (text or "").lower()
    cats = load_taxonomy()

    # Exact token match
    tokens = re.split(r"\W+", text)
    for c in cats:
        for a in c["aliases_norm"]:
            if a in tokens:
                return c["id"], "token"

    # Substring match
    for c in cats:
        for a in c["aliases_norm"]:
            if a in text:
                return c["id"], "substring"

    return None, None


def prelabel_df(df, text_col="transaction", label_col="label"):
    # For rows with empty label, attempt to prelabel using aliases
    preds = []
    for _, r in df.iterrows():
        if pd.isna(r.get(label_col)):
            lid, method = alias_lookup(r.get(text_col, "") or "")
            preds.append(lid)
        else:
            preds.append(r.get(label_col))

    df["_prelabel"] = preds
    df[label_col] = df[label_col].fillna(df["_prelabel"])
    return df
