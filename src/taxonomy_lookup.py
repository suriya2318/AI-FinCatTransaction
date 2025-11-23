import yaml
import re
from functools import lru_cache


@lru_cache()
def load_taxonomy(path="configs/taxonomy.yaml"):
    """
    Returns list of category dicts with keys: id, display_name, aliases_norm
    """
    with open(path, "r", encoding="utf8") as f:
        raw = yaml.safe_load(f)
    cats = raw.get("categories") or []
    for c in cats:
        c["display_name"] = c.get("display_name", c.get("id"))
        c["aliases_norm"] = [str(a).lower() for a in c.get("aliases", []) if a]
    return cats


def get_all_categories():
    """
    Return list of tuples (id, display_name) for UI lists.
    """
    cats = load_taxonomy()
    return [(c["id"], c["display_name"]) for c in cats]


def alias_lookup(text):
    """
    Checks the taxonomy aliases and returns a tuple:
      (category_id, method, matched_alias)
    method: 'token' for token-exact match (high precision), 'substring' for substring match (lower precision)
    matched_alias: the alias text that matched (lowercase)
    Returns (None, None, None) if no match.
    """
    text_l = (text or "").lower()
    if not text_l:
        return None, None, None

    tokens = re.split(r"\W+", text_l)
    cats = load_taxonomy()
    for c in cats:
        for a in c["aliases_norm"]:
            if a and a in tokens:
                return c["id"], "token", a
    for c in cats:
        for a in c["aliases_norm"]:
            if a and a in text_l:
                return c["id"], "substring", a

    return None, None, None
