"""Prepare the downloaded Project Gutenberg ebook as traceable corpus units."""
import json
import re
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SOURCE_PATH = DATA_DIR / "alice_adventures_in_wonderland.txt"
STORY_PATH = DATA_DIR / "alice_story_extract.txt"
UNITS_PATH = DATA_DIR / "alice_units.jsonl"
MANIFEST_PATH = DATA_DIR / "alice_manifest.json"
SOURCE_URL = "https://www.gutenberg.org/ebooks/11"
SOURCE_ID = "alice11"
TARGET_UNIT_CHARS = 3000
MIN_UNIT_CHARS = 1500

CHAPTER_PATTERN = re.compile(
    r"^CHAPTER (?P<number>I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII)\."
    r"[ \t]*\r?\n(?P<title>[^\r\n]+)",
    re.MULTILINE,
)


def extract_paragraphs(text: str) -> list[str]:
    paragraphs = []
    for raw_paragraph in re.split(r"\n\s*\n", text):
        paragraph = raw_paragraph.strip()
        if paragraph and not re.fullmatch(r"\[Illustration\]", paragraph, re.IGNORECASE):
            paragraphs.append(paragraph)
    return paragraphs


def group_paragraphs(paragraphs: list[str]) -> list[list[str]]:
    groups: list[list[str]] = []
    current: list[str] = []
    current_chars = 0

    for paragraph in paragraphs:
        current.append(paragraph)
        current_chars += len(paragraph) + (2 if len(current) > 1 else 0)
        if current_chars >= TARGET_UNIT_CHARS:
            groups.append(current)
            current = []
            current_chars = 0

    if current:
        if groups and sum(map(len, current)) + 2 * (len(current) - 1) < MIN_UNIT_CHARS:
            groups[-1].extend(current)
        else:
            groups.append(current)

    if any(len("\n\n".join(group)) < MIN_UNIT_CHARS for group in groups):
        raise ValueError("A chapter produced a unit shorter than 1,500 characters.")
    return groups


def main() -> None:
    source_text = SOURCE_PATH.read_text(encoding="utf-8-sig")
    start_match = re.search(r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK [^\n]+\*\*\*", source_text)
    end_match = re.search(r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK [^\n]+\*\*\*", source_text)
    if not start_match or not end_match or end_match.start() <= start_match.end():
        raise ValueError("Could not locate Project Gutenberg ebook boundaries.")

    ebook_text = source_text[start_match.end():end_match.start()]
    chapters = list(CHAPTER_PATTERN.finditer(ebook_text))
    if len(chapters) != 12:
        raise ValueError(f"Expected 12 chapter headings; found {len(chapters)}.")

    story_parts: list[str] = []
    units: list[dict[str, object]] = []
    cursor = 0

    for chapter_index, chapter_match in enumerate(chapters):
        chapter_number = chapter_match.group("number")
        chapter_title = chapter_match.group("title").strip()
        chapter_end = chapters[chapter_index + 1].start() if chapter_index + 1 < len(chapters) else len(ebook_text)
        paragraphs = extract_paragraphs(ebook_text[chapter_match.end():chapter_end])
        if not paragraphs:
            raise ValueError(f"Chapter {chapter_number} has no extractable text.")

        heading = f"CHAPTER {chapter_number}. {chapter_title}"
        story_parts.append(heading)
        cursor += len(heading) + 2
        chapter_body_start = cursor
        chapter_body = "\n\n".join(paragraphs)
        story_parts.append(chapter_body)

        position = chapter_body_start
        for unit_index, paragraph_group in enumerate(group_paragraphs(paragraphs), start=1):
            unit_text = "\n\n".join(paragraph_group)
            unit_start = position
            unit_end = unit_start + len(unit_text)
            units.append({
                "unit_id": f"alice-ch{chapter_index + 1:02d}-u{unit_index:02d}",
                "source_id": SOURCE_ID,
                "document_id": SOURCE_ID,
                "source": SOURCE_URL,
                "chapter": chapter_number,
                "section": chapter_title,
                "position_start": unit_start,
                "position_end": unit_end,
                "char_count": len(unit_text),
                "text": unit_text,
            })
            position = unit_end + 2

        cursor += len(chapter_body) + 2

    story_extract = "\n\n".join(story_parts)
    for unit in units:
        start = int(unit["position_start"])
        end = int(unit["position_end"])
        if story_extract[start:end] != unit["text"]:
            raise ValueError(f"Source offset verification failed for {unit['unit_id']}.")

    if len(units) < 30:
        raise ValueError(f"Corpus has {len(units)} units; at least 30 are required.")

    STORY_PATH.write_text(story_extract + "\n", encoding="utf-8")
    with UNITS_PATH.open("w", encoding="utf-8", newline="\n") as units_file:
        for unit in units:
            units_file.write(json.dumps(unit, ensure_ascii=False) + "\n")

    manifest = {
        "source_id": SOURCE_ID,
        "title": "Alice's Adventures in Wonderland",
        "author": "Lewis Carroll",
        "source_path_or_url": SOURCE_URL,
        "download_url": "https://www.gutenberg.org/cache/epub/11/pg11.txt",
        "local_source_path": "alice_adventures_in_wonderland.txt",
        "license_or_permission": (
            "The ebook states it is free to use in the United States and distributed "
            "under the Project Gutenberg License. Verify local law before reuse or "
            "redistribution outside the United States."
        ),
        "accessed_at": "2026-11-14",
        "snapshot": "Project Gutenberg ebook #11; most recently updated 2025-06-26",
        "normalization": "Applied only when starter/src/ingestion.py loads the units; this preparation script preserves source paragraph text.",
        "segmentation": (
            "Unnormalized source paragraphs are grouped in source order, targeting "
            "3,000 raw characters per unit. A final remainder shorter than 1,500 "
            "raw characters is merged into the preceding unit in that chapter. "
            "The ingestion stage verifies each normalized unit remains at least "
            "1,500 characters."
        ),
        "position_definition": (
            "Each unit's position_start and position_end are 0-based, half-open "
            "character offsets into alice_story_extract.txt before ingestion normalization; "
            "chapter headings are included in the extracted file but not in unit text."
        ),
        "metadata_schema": [
            {
                "field": "unit_id",
                "type": "string",
                "source": "Generated from the chapter index and unit index.",
                "purpose": "Unique traceable identifier for each grouped paragraph unit.",
            },
            {
                "field": "source_id",
                "type": "string",
                "source": "Constant alice11 for Project Gutenberg ebook #11.",
                "purpose": "Stable identity for the source work.",
            },
            {
                "field": "document_id",
                "type": "string",
                "source": "Set to source_id.",
                "purpose": "Document-level identifier used by ingestion and retrieval.",
            },
            {
                "field": "source",
                "type": "URL string",
                "source": "Project Gutenberg ebook landing page.",
                "purpose": "Source provenance for each unit.",
            },
            {
                "field": "chapter",
                "type": "string",
                "source": "Roman numeral from the source chapter heading.",
                "purpose": "Chapter-level provenance and metadata filtering.",
            },
            {
                "field": "section",
                "type": "string",
                "source": "Title from the source chapter heading.",
                "purpose": "Chapter-title provenance and metadata filtering.",
            },
            {
                "field": "position_start",
                "type": "integer",
                "source": "Calculated while building alice_story_extract.txt.",
                "purpose": "Inclusive 0-based source offset before ingestion normalization.",
            },
            {
                "field": "position_end",
                "type": "integer",
                "source": "position_start plus the raw unit text length.",
                "purpose": "Exclusive source offset before ingestion normalization.",
            },
            {
                "field": "char_count",
                "type": "integer",
                "source": "Length of the unnormalized unit text.",
                "purpose": "Raw unit-size validation and provenance.",
            },
            {
                "field": "text",
                "type": "string",
                "source": "Source paragraphs grouped in order without text normalization.",
                "purpose": "Raw content ingested and normalized by starter/src/ingestion.py.",
            },
        ],
        "raw_extracted_char_count": len(story_extract),
        "unit_count": len(units),
        "units_path": "alice_units.jsonl",
        "units": [
            {
                "unit_id": unit["unit_id"],
                "chapter": unit["chapter"],
                "position_start": unit["position_start"],
                "position_end": unit["position_end"],
                "char_count": unit["char_count"],
            }
            for unit in units
        ],
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Prepared {len(units)} units ({sum(int(unit['char_count']) for unit in units)} raw content characters).")
    print(f"Created {STORY_PATH.name}, {UNITS_PATH.name}, and {MANIFEST_PATH.name}.")


if __name__ == "__main__":
    main()
