from __future__ import annotations
from typing import Dict
import re

SECTION_SYNONYMS = {
    "skills": ["skills", "technical skills", "core skills", "tools", "technologies"],
    "experience": ["experience", "work experience", "professional experience", "employment"],
    "projects": ["projects", "project experience"],
    "education": ["education", "academics", "academic background"],
}

def detect_sections(resume_text: str) -> Dict[str, bool]:
    t = resume_text.lower()
    found: Dict[str, bool] = {}
    for section, variants in SECTION_SYNONYMS.items():
        found[section] = any(re.search(rf"\b{re.escape(v)}\b", t) for v in variants)
    return found

def section_score(found: Dict[str, bool]) -> int:
    score = 0
    for key in ["skills", "experience", "education"]:
        if found.get(key):
            score += 5
    return score
