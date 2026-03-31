from pathlib import Path
import chardet
import pdfplumber
from docx import Document
from bs4 import BeautifulSoup
from utils.logger import get_logger

logger = get_logger("ingestion")


def extract_text_from_pdf(path: str) -> str:
    """Extract text from PDF using pdfplumber. Returns empty string for image-only PDFs."""
    texts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                texts.append(text)
    return "\n\n".join(texts)


def extract_text_from_docx(path: str) -> str:
    doc = Document(path)
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())


def extract_text_from_txt(path: str) -> str:
    raw_bytes = Path(path).read_bytes()
    detected = chardet.detect(raw_bytes)
    encoding = detected.get("encoding", "utf-8") or "utf-8"
    return raw_bytes.decode(encoding, errors="replace")


def extract_text_from_html(path: str) -> str:
    raw_bytes = Path(path).read_bytes()
    detected = chardet.detect(raw_bytes)
    encoding = detected.get("encoding", "utf-8") or "utf-8"
    html = raw_bytes.decode(encoding, errors="replace")
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def ingest_file(path: str) -> dict:
    """
    Returns:
        {filename, file_type, raw_text, char_count, is_scanned_pdf}
    """
    p = Path(path)
    suffix = p.suffix.lower()
    filename = p.name
    is_scanned_pdf = False

    if suffix == ".pdf":
        raw_text = extract_text_from_pdf(path)
        if len(raw_text.strip()) < 200:
            is_scanned_pdf = True
            logger.warning(f"{filename}: Very little text extracted — may be scanned PDF")
        file_type = "pdf"
    elif suffix in (".docx", ".doc"):
        raw_text = extract_text_from_docx(path)
        file_type = "docx"
    elif suffix in (".txt", ".md"):
        raw_text = extract_text_from_txt(path)
        file_type = "txt"
    elif suffix in (".html", ".htm"):
        raw_text = extract_text_from_html(path)
        file_type = "html"
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    return {
        "filename": filename,
        "file_type": file_type,
        "raw_text": raw_text,
        "char_count": len(raw_text),
        "is_scanned_pdf": is_scanned_pdf,
    }
