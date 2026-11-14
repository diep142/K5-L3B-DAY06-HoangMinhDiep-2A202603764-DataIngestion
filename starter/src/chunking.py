"""Fixed-size and paragraph-aware chunkers that preserve source metadata."""
import hashlib
import re

from .models import Chunk, DocumentPage


def _make_chunk(page: DocumentPage, text: str, mode: str, index: int) -> Chunk:
    identity = "\0".join((mode, page.document_id, str(page.page), str(index), text))
    chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
    return Chunk(
        chunk_id=chunk_id,
        document_id=page.document_id,
        source=page.source,
        page=page.page,
        text=text,
        metadata=dict(page.metadata),
    )


def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    """Split a page into fixed-length windows, avoiding tiny trailing fragments."""
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be non-negative and smaller than size")
    if not page.text:
        return []
    step = size - overlap
    starts = [0]
    minimum_tail = max(overlap, size // 4)
    maximum_overlap = max(overlap, size // 2)
    while starts[-1] + size < len(page.text):
        next_start = starts[-1] + step
        if len(page.text) - next_start < minimum_tail and len(starts) > 1:
            adjusted_start = len(page.text) - size
            previous_start = starts[-2]
            earliest_allowed_start = previous_start + size - maximum_overlap
            if adjusted_start >= earliest_allowed_start:
                starts[-1] = adjusted_start
                break
        starts.append(next_start)
    return [
        _make_chunk(page, window, "fixed", index)
        for index, start in enumerate(starts)
        if (window := page.text[start:start + size].strip())
    ]


def _split_long_paragraph(paragraph: str, max_size: int) -> list[str]:
    pieces: list[str] = []
    remaining = paragraph.strip()
    while len(remaining) > max_size:
        boundary = remaining.rfind(" ", 0, max_size + 1)
        if boundary <= 0:
            boundary = max_size
        pieces.append(remaining[:boundary].strip())
        remaining = remaining[boundary:].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    """Group whole paragraphs up to max_size and split oversized paragraphs."""
    if max_size <= 0:
        raise ValueError("max_size must be positive")
    if not page.text:
        return []
    paragraphs = [
        re.sub(r"\s+", " ", paragraph).strip()
        for paragraph in re.split(r"\n\s*\n", page.text)
        if paragraph.strip()
    ]

    chunks: list[Chunk] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            chunks.append(_make_chunk(page, "\n\n".join(current), "paragraph", len(chunks)))
            current.clear()

    for paragraph in paragraphs:
        for piece in _split_long_paragraph(paragraph, max_size):
            if current and len("\n\n".join([*current, piece])) > max_size:
                flush()
            current.append(piece)
    flush()
    return chunks
