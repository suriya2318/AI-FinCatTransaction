# src/taxonomy_lookup.py
import yaml
import re
from functools import lru_cache
import os

DEFAULT_PATH = os.path.join("configs", "taxonomy.yaml")


@lru_cache()
def load_taxonomy(path=DEFAULT_PATH):
    """Load taxonomy YAML and return list of category dicts."""
    with open(path, "r", encoding="utf8") as f:
        tax = yaml.safe_load(f)
    categories = tax.get("categories", [])
    # normalize aliases and ensure fields
    for c in categories:
        c["id"] = str(c.get("id"))
        c["display_name"] = c.get("display_name", c["id"])
        c["aliases_norm"] = [
            a.lower() for a in c.get("aliases", []) if isinstance(a, str)
        ]
    return categories


def alias_lookup(text, path=DEFAULT_PATH):
    """
    Return (category_id, method) where method:
      - 'token' for high-precision whole-word match
      - 'substring' for conservative substring (aliases length >=4)
      - (None, None) if no match
    """
    if not text:
        return None, None
    text_l = text.lower()
    cats = load_taxonomy(path)
    # high-precision: whole-word matches
    for c in cats:
        for a in c["aliases_norm"]:
            if not a:
                continue
            # word boundary matching
            pattern = r"\b" + re.escape(a) + r"\b"
            if re.search(pattern, text_l):
                return c["id"], "token"
    # conservative substring match (only for aliases length >= 4 to avoid short collisions)
    for c in cats:
        for a in c["aliases_norm"]:
            if len(a) >= 4 and a in text_l:
                return c["id"], "substring"
    return None, None


def get_all_categories(path=DEFAULT_PATH):
    """Return list of tuples (id, display_name) for UI rendering"""
    cats = load_taxonomy(path)
    return [(c["id"], c["display_name"]) for c in cats]
