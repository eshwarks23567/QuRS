import io
from pathlib import Path

import pdfplumber


def load_pdf(file_path: str) -> str:
    """Extract all text from a PDF file at the given path."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")
    if not path.suffix.lower() == ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {file_path}")

    return _extract_from_plumber(str(path))


def load_pdf_bytes(content: bytes) -> str:
    """Extract text from raw PDF bytes (e.g., from an HTTP upload)."""
    return _extract_from_plumber(io.BytesIO(content))


def _extract_from_plumber(source) -> str:
    """Internal helper: opens source (path str or BytesIO) and extracts page text."""
    pages: list[str] = []
    with pdfplumber.open(source) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text and text.strip():
                pages.append(text.strip())
    if not pages:
        raise ValueError("No extractable text found in the PDF.")
    return "\n\n".join(pages)
