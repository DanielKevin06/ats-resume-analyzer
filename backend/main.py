from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import pdfplumber
from docx import Document
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# FastAPI app (IMPORTANT: must be named "app")
# -----------------------------
app = FastAPI(title="Resume ATS Analyzer API", version="1.0.0")

# Allow Streamlit to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Response model
# -----------------------------
class AnalyzeResponse(BaseModel):
    total_score: int
    similarity_score: int
    keyword_score: int
    section_score: int
    formatting_score: int
    similarity: float

    keywords_total: int
    keywords_matched: List[str]
    keywords_missing: List[str]
    keywords_missing_grouped: Dict[str, List[str]]

    sections_found: Dict[str, bool]
    warnings: List[str]
    action_items: List[str]

    resume_text: str


# -----------------------------
# Helpers
# -----------------------------
SECTION_HEADINGS = {
    "summary": ["summary", "professional summary", "profile", "about"],
    "skills": ["skills", "technical skills", "core skills"],
    "experience": ["experience", "work experience", "professional experience", "employment"],
    "projects": ["projects", "academic projects", "personal projects"],
    "education": ["education"],
    "certifications": ["certifications", "certificates", "certification"],
}

STOPWORDS_MINI = {
    "and", "or", "the", "a", "an", "to", "of", "in", "for", "with", "on", "at",
    "is", "are", "as", "by", "be", "from", "this", "that", "you", "we", "will",
    "your", "our", "their", "they", "it", "etc"
}


def _clean_text(t: str) -> str:
    t = t.replace("\x00", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def extract_text_from_pdf(file_bytes: bytes) -> str:
    text_parts = []
    with pdfplumber.open(io_bytes(file_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if text.strip():
                text_parts.append(text)
    return _clean_text("\n".join(text_parts))


def extract_text_from_docx(file_bytes: bytes) -> str:
    doc = Document(io_bytes(file_bytes))
    paras = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    return _clean_text("\n".join(paras))


def io_bytes(b: bytes):
    import io
    return io.BytesIO(b)


def get_resume_text(upload: UploadFile) -> str:
    content = upload.file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    name = (upload.filename or "").lower()
    ctype = (upload.content_type or "").lower()

    try:
        if name.endswith(".pdf") or "pdf" in ctype:
            return extract_text_from_pdf(content)
        if name.endswith(".docx") or "word" in ctype or "officedocument" in ctype:
            return extract_text_from_docx(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read resume file: {e}")

    raise HTTPException(status_code=400, detail="Unsupported file type. Upload PDF or DOCX.")


def compute_similarity(resume_text: str, jd_text: str) -> float:
    # TF-IDF cosine similarity in [0..1]
    vect = TfidfVectorizer(stop_words="english")
    X = vect.fit_transform([resume_text, jd_text])
    sim = cosine_similarity(X[0:1], X[1:2])[0][0]
    if sim != sim:  # NaN guard
        return 0.0
    return float(max(0.0, min(1.0, sim)))


def extract_keywords_from_jd(jd_text: str, max_features: int = 80) -> List[str]:
    jd_text = jd_text.lower()

    # CountVectorizer ngrams gives us phrases (good for ATS)
    vec = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=max_features,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\+\#\.\-]{1,}\b",
    )
    X = vec.fit_transform([jd_text])
    terms = vec.get_feature_names_out().tolist()

    # Light cleanup
    cleaned = []
    for t in terms:
        tt = t.strip().lower()
        if len(tt) < 2:
            continue
        if tt in STOPWORDS_MINI:
            continue
        # filter junk
        if tt.isdigit():
            continue
        cleaned.append(tt)

    # unique, keep order
    seen = set()
    final = []
    for k in cleaned:
        if k not in seen:
            final.append(k)
            seen.add(k)

    return final


def match_keywords(resume_text: str, keywords: List[str]) -> (List[str], List[str]):
    rlow = resume_text.lower()
    matched = []
    missing = []
    for k in keywords:
        # exact substring match (simple + fast + effective)
        if k in rlow:
            matched.append(k)
        else:
            missing.append(k)
    return matched, missing


def detect_sections(resume_text: str) -> Dict[str, bool]:
    t = resume_text.lower()
    found = {}
    for section, variants in SECTION_HEADINGS.items():
        found[section] = any(v in t for v in variants)
    return found


def score_sections(found: Dict[str, bool]) -> int:
    # section score out of 15
    # (summary, skills, experience, projects, education) are most important
    weights = {
        "summary": 3,
        "skills": 3,
        "experience": 4,
        "projects": 3,
        "education": 2,
        "certifications": 0,  # bonus handled elsewhere
    }
    score = sum(weights[s] for s, ok in found.items() if ok and s in weights)
    return int(min(15, max(0, score)))


def score_keywords(matched: List[str], total: int) -> int:
    # keyword score out of 45
    if total <= 0:
        return 0
    ratio = len(matched) / total
    return int(round(45 * ratio))


def score_similarity(sim: float) -> int:
    # similarity score out of 25
    return int(round(25 * max(0.0, min(1.0, sim))))


def score_formatting(resume_text: str) -> (int, List[str]):
    # formatting score out of 15 (simple heuristics)
    warnings = []
    score = 15

    if len(resume_text) < 400:
        score -= 5
        warnings.append("Resume text looks very short. ATS may not extract enough content.")

    # if it contains too many weird symbols
    bad_chars = len(re.findall(r"[■◆●►•]{6,}", resume_text))
    if bad_chars > 0:
        score -= 2
        warnings.append("Heavy special bullet symbols detected. Use simple bullets for ATS safety.")

    # if no dates present
    if not re.search(r"\b(19|20)\d{2}\b", resume_text):
        score -= 2
        warnings.append("No year/date pattern detected. Ensure experience dates are visible to ATS.")

    return int(max(0, min(15, score))), warnings


def group_missing_keywords(missing: List[str]) -> Dict[str, List[str]]:
    # Simple grouping:
    # Must-have: first 15
    # Nice-to-have: next 20
    # Role-specific: remaining
    return {
        "must_have": missing[:15],
        "nice_to_have": missing[15:35],
        "role_specific": missing[35:60],
    }


def build_action_items(
    section_found: Dict[str, bool],
    missing: List[str],
    sim: float,
    fmt_warnings: List[str],
) -> List[str]:
    items = []

    if sim < 0.35:
        items.append("Tailor your Summary + 2–3 bullets to mirror the JD wording (tools, responsibilities, role title).")

    if not section_found.get("skills", False):
        items.append("Add a clear 'Skills' section with tools, languages, frameworks, and platforms.")

    if not section_found.get("projects", False):
        items.append("Add a 'Projects' section with 2–3 projects and measurable impact (metrics/results).")

    if not section_found.get("experience", False):
        items.append("Add 'Experience' section (even internships/part-time/volunteer) with action + metric bullets.")

    if missing:
        items.append(f"Add missing keywords you truly have (Skills/Projects/Experience): {', '.join(missing[:8])}.")

    # include 1–2 formatting warnings as action items
    for w in fmt_warnings[:2]:
        items.append(w)

    # unique
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health() -> str:
    return "ok"


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    resume: UploadFile = File(...),
    # accept BOTH names: jd (streamlit) and jd_text (swagger)
    jd: Optional[str] = Form(None),
    jd_text: Optional[str] = Form(None),
):
    job_text = (jd_text or jd or "").strip()
    if not job_text:
        raise HTTPException(status_code=422, detail="Job Description is required (jd or jd_text).")

    resume_text = get_resume_text(resume)
    if not resume_text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from resume. Try another PDF/DOCX.")

    sim = compute_similarity(resume_text, job_text)
    keywords = extract_keywords_from_jd(job_text, max_features=80)
    matched, missing = match_keywords(resume_text, keywords)

    sections_found = detect_sections(resume_text)

    keyword_score = score_keywords(matched, len(keywords))          # /45
    similarity_score = score_similarity(sim)                        # /25
    section_score = score_sections(sections_found)                  # /15
    formatting_score, fmt_warnings = score_formatting(resume_text)  # /15

    total = int(keyword_score + similarity_score + section_score + formatting_score)
    total = int(max(0, min(100, total)))

    grouped = group_missing_keywords(missing)
    warnings = fmt_warnings.copy()

    # extra warning: missing important sections
    if not sections_found.get("education", False):
        warnings.append("Education section not detected. Add 'Education' heading clearly.")
    if not sections_found.get("skills", False):
        warnings.append("Skills section not detected. Add a clear 'Skills' heading.")

    action_items = build_action_items(sections_found, missing, sim, fmt_warnings)

    return AnalyzeResponse(
        total_score=total,
        similarity_score=similarity_score,
        keyword_score=keyword_score,
        section_score=section_score,
        formatting_score=formatting_score,
        similarity=sim,
        keywords_total=len(keywords),
        keywords_matched=matched,
        keywords_missing=missing,
        keywords_missing_grouped=grouped,
        sections_found=sections_found,
        warnings=warnings,
        action_items=action_items,
        resume_text=resume_text,
    )
