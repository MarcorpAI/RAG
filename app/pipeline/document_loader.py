from io import BytesIO
from pathlib import Path


SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


class UnsupportedDocumentType(ValueError):
    pass


class EmptyDocumentError(ValueError):
    pass


def extract_text(filename: str, content: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnsupportedDocumentType("Only PDF and TXT files are supported")

    if suffix == ".txt":
        text = content.decode("utf-8", errors="replace")
    else:
        text = _extract_pdf_text(content)

    normalized = " ".join(text.split())
    if not normalized:
        raise EmptyDocumentError("Document did not contain extractable text")
    return normalized


def _extract_pdf_text(content: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("pypdf is required to parse PDF uploads") from exc

    reader = PdfReader(BytesIO(content))
    return "\n".join(page.extract_text() or "" for page in reader.pages)
