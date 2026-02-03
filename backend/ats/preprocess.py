from __future__ import annotations
import re

_ALLOWED_PUNCT = r"\+\#"

def normalize_text(text: str) -> str:
    text = text.replace("\u00A0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def clean_for_matching(text: str) -> str:
    t = text.lower()
    t = normalize_text(t)
    t = re.sub(rf"[^a-z0-9\s{_ALLOWED_PUNCT}]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t
