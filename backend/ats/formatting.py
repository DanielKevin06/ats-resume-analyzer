from __future__ import annotations
import re
from dataclasses import dataclass

@dataclass
class FormattingCheck:
    formatting_score: int
    warnings: list[str]

def formatting_heuristics(raw_resume_text: str, word_count: int) -> FormattingCheck:
    score = 10
    warnings: list[str] = []

    if word_count > 900:
        score -= 3
        warnings.append("Resume seems long (word count is high). Consider compressing to ~1–2 pages.")
    elif word_count < 200:
        score -= 2
        warnings.append("Resume seems very short. Consider adding more detail to projects/experience.")

    bullets = re.findall(r"(^|\n)\s*[\-\u2022\*]\s+", raw_resume_text)
    if len(bullets) < 5:
        score -= 2
        warnings.append("Low bullet count detected. ATS-friendly resumes usually use bullet points for achievements.")

    special = re.findall(r"[•★✓✅❖▶️→⇢▪️]", raw_resume_text)
    if len(special) > 10:
        score -= 1
        warnings.append("Too many decorative symbols/icons detected. Some ATS parsers lose text around icons.")

    tabley = re.findall(r"(\|{2,}|_{5,}|-{8,})", raw_resume_text)
    if len(tabley) > 3:
        score -= 2
        warnings.append("Possible tables/columns detected. ATS can struggle with multi-column layouts or tables.")

    score = max(0, min(10, score))
    return FormattingCheck(formatting_score=score, warnings=warnings)
