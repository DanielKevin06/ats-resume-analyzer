from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .preprocess import clean_for_matching
from .keywords import extract_keywords_from_jd
from .sections import detect_sections, section_score
from .formatting import formatting_heuristics

@dataclass
class ATSResult:
    total_score: int
    similarity_score: int
    keyword_score: int
    section_score: int
    formatting_score: int
    similarity: float
    keywords_total: int
    keywords_matched: List[str]
    keywords_missing: List[str]
    sections_found: Dict[str, bool]
    warnings: List[str]

def _cosine_sim(a: str, b: str) -> float:
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=5000)
    X = vec.fit_transform([a, b])
    sim = cosine_similarity(X[0:1], X[1:2])[0][0]
    return float(sim)

def analyze_resume_against_jd(raw_resume_text: str, resume_word_count: int, jd_text: str) -> ATSResult:
    resume_clean = clean_for_matching(raw_resume_text)
    jd_clean = clean_for_matching(jd_text)

    sim = _cosine_sim(resume_clean, jd_clean)
    similarity_points = int(round(sim * 40))

    jd_keywords = extract_keywords_from_jd(jd_text, max_keywords=60)

    matched = []
    missing = []
    for kw in jd_keywords:
        if kw and kw in resume_clean:
            matched.append(kw)
        else:
            missing.append(kw)

    total_kw = max(1, len(jd_keywords))
    match_ratio = len(matched) / total_kw
    keyword_points = int(round(match_ratio * 35))

    sections = detect_sections(raw_resume_text)
    section_points = section_score(sections)

    fmt = formatting_heuristics(raw_resume_text, resume_word_count)

    total = similarity_points + keyword_points + section_points + fmt.formatting_score
    total = max(0, min(100, total))

    return ATSResult(
        total_score=total,
        similarity_score=similarity_points,
        keyword_score=keyword_points,
        section_score=section_points,
        formatting_score=fmt.formatting_score,
        similarity=sim,
        keywords_total=len(jd_keywords),
        keywords_matched=sorted(set(matched)),
        keywords_missing=sorted(set(missing)),
        sections_found=sections,
        warnings=fmt.warnings,
    )
