from __future__ import annotations
from typing import List, Set
import re
import spacy

_nlp = None

def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp

COMMON_SKILL_PATTERNS = [
    r"machine learning",
    r"deep learning",
    r"data science",
    r"data engineering",
    r"data mining",
    r"natural language processing",
    r"computer vision",
    r"information security",
    r"android security",
    r"penetration testing",
]

def extract_keywords_from_jd(jd_text: str, max_keywords: int = 60) -> List[str]:
    nlp = _get_nlp()
    doc = nlp(jd_text)

    keywords: Set[str] = set()

    jd_lower = jd_text.lower()
    for pat in COMMON_SKILL_PATTERNS:
        if re.search(pat, jd_lower):
            keywords.add(pat)

    tech_like = re.findall(r"\b[A-Z][A-Z0-9\+\#]{1,}\b", jd_text)
    for t in tech_like:
        keywords.add(t.lower())

    tools = re.findall(
        r"\b(Python|Java|C\+\+|C\#|SQL|NoSQL|AWS|Azure|GCP|Docker|Kubernetes|Spark|Hadoop|Kafka|TensorFlow|PyTorch|scikit-learn|Linux)\b",
        jd_text
    )
    for t in tools:
        keywords.add(t.lower())

    for chunk in doc.noun_chunks:
        c = chunk.text.strip().lower()
        if len(c) < 3:
            continue
        if c in {"experience", "skills", "requirements", "responsibilities"}:
            continue
        if sum(1 for tok in chunk if tok.is_stop) / max(1, len(chunk)) > 0.6:
            continue
        c = re.sub(r"[^a-z0-9\s\+\#]", " ", c)
        c = re.sub(r"\s+", " ", c).strip()
        if 3 <= len(c) <= 45:
            keywords.add(c)

    for tok in doc:
        if tok.is_stop or tok.is_punct or tok.like_num:
            continue
        if tok.pos_ in {"PROPN", "NOUN"} and len(tok.text) >= 3:
            t = tok.text.lower()
            t = re.sub(r"[^a-z0-9\+\#]", "", t)
            if t:
                keywords.add(t)

    sorted_kw = sorted(keywords, key=lambda x: (-len(x.split()), -len(x), x))
    return sorted_kw[:max_keywords]
