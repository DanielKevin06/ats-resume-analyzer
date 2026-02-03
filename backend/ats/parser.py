from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import io
import re

import pdfplumber
from docx import Document


@dataclass
class ParsedResume:
    text: str
    word_count: int
    char_count: int


def _basic_counts(text: str) -> Tuple[int, int]:
    words = re.findall(r"\b\w+\b", text)
    return (len(words), len(text))


def extract_text_from_pdf(file_bytes: bytes) -> ParsedResume:
    text_parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
    text = "\n".join(text_parts).strip()
    wc, cc = _basic_counts(text)
    return ParsedResume(text=text, word_count=wc, char_count=cc)


def extract_text_from_docx(file_bytes: bytes) -> ParsedResume:
    doc = Document(io.BytesIO(file_bytes))
    text_parts = [p.text for p in doc.paragraphs if p.text]
    text = "\n".join(text_parts).strip()
    wc, cc = _basic_counts(text)
    return ParsedResume(text=text, word_count=wc, char_count=cc)


def extract_resume_text(filename: str, file_bytes: bytes) -> ParsedResume:
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    if lower.endswith(".docx"):
        return extract_text_from_docx(file_bytes)
    raise ValueError("Unsupported file type. Please upload PDF or DOCX.")
