import unittest

from src.chunking import by_paragraph, fixed_size
from src.ingestion import normalize_text
from src.models import DocumentPage


class IngestionAndChunkingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.page = DocumentPage(
            document_id="alice11",
            source="https://example.test/book",
            page="unit-1",
            text="First paragraph with words.\n\nSecond paragraph has more words.",
            metadata={"unit_id": "unit-1", "chapter": "I"},
        )

    def test_normalize_text_preserves_paragraphs_and_normalizes_unicode(self) -> None:
        self.assertEqual(normalize_text("Cafe\u0301\u00a0  text\r\n\r\nnext"), "Café text\n\nnext")

    def test_fixed_size_chunks_overlap_and_keep_trace_metadata(self) -> None:
        chunks = fixed_size(self.page, size=20, overlap=5)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk.text) <= 20 for chunk in chunks))
        self.assertTrue(all(chunk.metadata["unit_id"] == "unit-1" for chunk in chunks))
        self.assertTrue(all(chunk.page == "unit-1" for chunk in chunks))
        self.assertEqual(len({chunk.chunk_id for chunk in chunks}), len(chunks))

    def test_fixed_size_avoids_a_tiny_trailing_fragment(self) -> None:
        page = DocumentPage("alice11", self.page.source, 1, "x" * 76)

        chunks = fixed_size(page, size=40, overlap=5)

        self.assertEqual([len(chunk.text) for chunk in chunks], [40, 40])

    def test_paragraph_chunking_respects_limit_and_source_metadata(self) -> None:
        page = DocumentPage(
            document_id=self.page.document_id,
            source=self.page.source,
            page=self.page.page,
            text="First paragraph.\n\nThis paragraph is deliberately longer than ten characters.",
            metadata=self.page.metadata,
        )
        chunks = by_paragraph(page, max_size=20)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk.text) <= 20 for chunk in chunks))
        self.assertTrue(all(chunk.metadata["chapter"] == "I" for chunk in chunks))

    def test_invalid_chunk_parameters_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            fixed_size(self.page, size=0, overlap=0)
        with self.assertRaises(ValueError):
            fixed_size(self.page, size=10, overlap=10)
        with self.assertRaises(ValueError):
            by_paragraph(self.page, max_size=0)


if __name__ == "__main__":
    unittest.main()
