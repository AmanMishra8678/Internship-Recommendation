"""
resume_parser.py
----------------
PDF resume parsing and NLP-based skill extraction.
Uses PyMuPDF (fitz) for reliable text extraction with fallback to pdfminer.
"""

import re
import io
from typing import Tuple, List, Dict

from modules.preprocessor import extract_skills_from_text, clean_text


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract raw text from a PDF given its binary content.
    Tries PyMuPDF first (faster), falls back to pdfminer.six.

    Parameters
    ----------
    file_bytes : raw bytes of the uploaded PDF file

    Returns
    -------
    Extracted plain text string.
    """
    text = _extract_with_pymupdf(file_bytes)
    if not text.strip():
        text = _extract_with_pdfminer(file_bytes)
    return text


def _extract_with_pymupdf(file_bytes: bytes) -> str:
    """Primary extraction using PyMuPDF (fitz)."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages = [page.get_text("text") for page in doc]
        doc.close()
        return "\n".join(pages)
    except Exception as e:
        print(f"[PyMuPDF] Extraction failed: {e}")
        return ""


def _extract_with_pdfminer(file_bytes: bytes) -> str:
    """Fallback extraction using pdfminer.six."""
    try:
        from pdfminer.high_level import extract_text_to_fp
        from pdfminer.layout import LAParams
        output = io.StringIO()
        extract_text_to_fp(
            io.BytesIO(file_bytes),
            output,
            laparams=LAParams(),
            output_type="text",
            codec="utf-8",
        )
        return output.getvalue()
    except Exception as e:
        print(f"[pdfminer] Extraction failed: {e}")
        return ""


# ---------------------------------------------------------------------------
# Resume section parsing
# ---------------------------------------------------------------------------

# Section header patterns (case-insensitive)
_SECTION_HEADERS = {
    "skills":       r"(technical\s+)?skills",
    "experience":   r"(work\s+)?experience|employment|projects?\s+experience",
    "education":    r"education|academic",
    "projects":     r"projects?",
    "summary":      r"summary|objective|profile|about",
    "certifications": r"certifications?|courses?|achievements?",
}


def parse_resume_sections(text: str) -> Dict[str, str]:
    """
    Split resume text into logical sections using regex-based header detection.

    Returns a dict like:
    {
        "skills": "Python, Machine Learning ...",
        "experience": "...",
        "education": "...",
        ...
        "full_text": "<all text>"
    }
    """
    sections: Dict[str, str] = {"full_text": text}

    # Build a combined pattern to detect any section header
    header_pattern = re.compile(
        r"^\s*(" + "|".join(_SECTION_HEADERS.values()) + r")\s*[:\-]?\s*$",
        re.IGNORECASE | re.MULTILINE,
    )

    lines = text.split("\n")
    current_section = "summary"
    buffer: List[str] = []
    section_content: Dict[str, List[str]] = {k: [] for k in _SECTION_HEADERS}
    section_content["summary"] = []

    for line in lines:
        matched_section = None
        for section_name, pattern in _SECTION_HEADERS.items():
            if re.fullmatch(r"\s*" + pattern + r"\s*[:\-]?", line.strip(), re.IGNORECASE):
                matched_section = section_name
                break

        if matched_section:
            # Save buffered lines to the previous section
            section_content[current_section].extend(buffer)
            buffer = []
            current_section = matched_section
        else:
            buffer.append(line)

    # Flush remaining buffer
    section_content[current_section].extend(buffer)

    # Convert lists to strings
    for section_name, lines_list in section_content.items():
        content = "\n".join(lines_list).strip()
        if content:
            sections[section_name] = content

    return sections


def extract_contact_info(text: str) -> Dict[str, str]:
    """
    Extract email addresses and phone numbers from resume text.
    Returns a dict with 'email' and 'phone' keys.
    """
    info: Dict[str, str] = {}

    # Email
    email_match = re.search(r"[\w.\-]+@[\w.\-]+\.\w{2,}", text)
    if email_match:
        info["email"] = email_match.group(0)

    # Phone (Indian / international formats)
    phone_match = re.search(
        r"(\+?91[\-\s]?)?[6-9]\d{9}|(\+?[1-9]\d{1,14})", text
    )
    if phone_match:
        info["phone"] = phone_match.group(0)

    return info


def parse_resume(file_bytes: bytes) -> Dict:
    """
    Full resume parsing pipeline.

    Returns
    -------
    {
        "raw_text"   : <full extracted text>,
        "sections"   : { "skills": ..., "experience": ..., ... },
        "skills"     : ["python", "machine learning", ...],
        "contact"    : { "email": ..., "phone": ... },
        "word_count" : int,
    }
    """
    raw_text = extract_text_from_pdf(file_bytes)

    if not raw_text.strip():
        return {
            "raw_text": "",
            "sections": {},
            "skills": [],
            "contact": {},
            "word_count": 0,
            "error": "Could not extract text from PDF. The file may be scanned or image-based.",
        }

    sections = parse_resume_sections(raw_text)

    # Prioritise the 'skills' section for extraction; fall back to full text
    skill_source = sections.get("skills", raw_text)
    extracted_skills = extract_skills_from_text(skill_source)

    # If skills section is sparse, supplement from full text
    if len(extracted_skills) < 5:
        extracted_skills = extract_skills_from_text(raw_text)

    contact = extract_contact_info(raw_text)

    return {
        "raw_text": raw_text,
        "sections": {k: v for k, v in sections.items() if k != "full_text"},
        "skills": extracted_skills,
        "contact": contact,
        "word_count": len(raw_text.split()),
    }
