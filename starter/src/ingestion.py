"""Read PDF or traceable text units and normalize their content."""
import json
import logging
import re
import unicodedata
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from .models import DocumentPage

logger = logging.getLogger(__name__)


def load_pdf(path: Path, document_id: str, source: str) -> list[DocumentPage]:
    """Extract every physical page and record pages with no readable text."""
    reader = PdfReader(str(path))
    result: list[DocumentPage] = []
    for page_number, pdf_page in enumerate(reader.pages, start=1):
        metadata: dict[str, object] = {}
        try:
            text = normalize_text(pdf_page.extract_text() or "")
        except (PdfReadError, ValueError) as error:
            logger.exception("Could not extract page %d from %s", page_number, path)
            text = ""
            metadata["extraction_error"] = f"{type(error).__name__}: {error}"
        if not text:
            logger.warning("Page %d of %s has no extractable text", page_number, path)
            metadata["empty_text"] = True
        result.append(DocumentPage(document_id, source, page_number, text, metadata))
    return result


def load_text_units(path: Path) -> list[DocumentPage]:
    """Load prepared JSONL units and keep the unit ID as the page equivalent."""
    result: list[DocumentPage] = []
    with path.open(encoding="utf-8") as source_file:
        for line_number, line in enumerate(source_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                required = (
                    "document_id", "source", "unit_id", "chapter", "section",
                    "position_start", "position_end", "char_count", "text",
                )
                missing = [field for field in required if field not in record]
                if missing:
                    raise ValueError(f"missing fields: {', '.join(missing)}")
                normalized_text = normalize_text(str(record["text"]))
                result.append(DocumentPage(
                    document_id=str(record["document_id"]),
                    source=str(record["source"]),
                    page=str(record["unit_id"]),
                    text=normalized_text,
                    metadata={
                        "unit_id": str(record["unit_id"]),
                        "chapter": str(record["chapter"]),
                        "section": str(record["section"]),
                        "position_start": int(record["position_start"]),
                        "position_end": int(record["position_end"]),
                        "char_count": int(record["char_count"]),
                        "normalized_char_count": len(normalized_text),
                    },
                ))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid source unit at {path}:{line_number}: {error}") from error
    if not result:
        raise ValueError(f"No source units found in {path}")
    return result


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace while preserving paragraph breaks."""
    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\u00a0", " ")
    normalized = re.sub(r"[^\S\n]+", " ", normalized)
    normalized = re.sub(r" *\n *", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    normalized = re.sub(r"(?<!\n)\n(?!\n)", " ", normalized)
    return normalized.strip()
