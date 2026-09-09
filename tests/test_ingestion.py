"""
Unit tests for Phase 2: PDF Ingestion and Document Chunking.
"""

import tempfile
import unittest
from pathlib import Path
from langchain_core.documents import Document
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from ingestion.pdf_loader import load_pdf
from ingestion.text_splitter import split_documents, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP


class TestPdfLoader(unittest.TestCase):
    """Tests for the PDF loader module."""

    def setUp(self) -> None:
        """Create a temporary PDF file for testing within test scope."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.pdf_path = Path(self.temp_dir.name) / "test_document.pdf"

        # Generate a 2-page test PDF
        c = canvas.Canvas(str(self.pdf_path), pagesize=letter)
        c.drawString(100, 750, "First page content for PDF ingestion testing.")
        c.showPage()
        c.drawString(100, 750, "Second page content for PDF ingestion testing.")
        c.showPage()
        c.save()

    def tearDown(self) -> None:
        """Clean up temporary directory and files."""
        self.temp_dir.cleanup()

    def test_load_pdf_success(self) -> None:
        """Test that load_pdf successfully loads a valid PDF and extracts documents with metadata."""
        documents = load_pdf(self.pdf_path)

        self.assertEqual(len(documents), 2)
        self.assertIn("First page content", documents[0].page_content)
        self.assertIn("Second page content", documents[1].page_content)

        # Verify page-related metadata preservation
        self.assertEqual(documents[0].metadata.get("page"), 0)
        self.assertEqual(documents[1].metadata.get("page"), 1)
        self.assertEqual(Path(documents[0].metadata.get("source")).name, "test_document.pdf")

    def test_load_pdf_missing_file_raises_error(self) -> None:
        """Test that load_pdf raises FileNotFoundError for non-existent file."""
        non_existent_path = Path(self.temp_dir.name) / "does_not_exist.pdf"

        with self.assertRaises(FileNotFoundError) as ctx:
            load_pdf(non_existent_path)

        self.assertIn("PDF file not found", str(ctx.exception))

    def test_load_pdf_directory_path_raises_error(self) -> None:
        """Test that load_pdf raises ValueError when given a directory path."""
        with self.assertRaises(ValueError) as ctx:
            load_pdf(self.temp_dir.name)

        self.assertIn("Path is not a file", str(ctx.exception))


class TestTextSplitter(unittest.TestCase):
    """Tests for the text splitter module."""

    def test_split_documents_produces_multiple_chunks(self) -> None:
        """Test that split_documents produces multiple chunks from sufficiently long text."""
        long_content = (
            "Artificial Intelligence and Document Intelligence enable automated "
            "extraction, processing, and understanding of content from enterprise documents. "
        ) * 30  # ~3000 characters

        doc = Document(page_content=long_content, metadata={"source": "test.pdf", "page": 0})
        chunks = split_documents([doc], chunk_size=500, chunk_overlap=50)

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertIsInstance(chunk, Document)
            self.assertLessEqual(len(chunk.page_content), 550)

    def test_split_documents_preserves_metadata(self) -> None:
        """Test that text splitting preserves document metadata across all generated chunks."""
        long_content = "Important document segment with specific details. " * 50
        metadata = {
            "source": "contract.pdf",
            "page": 3,
            "category": "legal",
        }
        doc = Document(page_content=long_content, metadata=metadata)

        chunks = split_documents([doc], chunk_size=300, chunk_overlap=50)

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertEqual(chunk.metadata.get("source"), "contract.pdf")
            self.assertEqual(chunk.metadata.get("page"), 3)
            self.assertEqual(chunk.metadata.get("category"), "legal")

    def test_split_documents_defaults(self) -> None:
        """Test that default chunk_size and chunk_overlap are used when not specified."""
        # Content shorter than default chunk size (1000) produces 1 chunk
        short_content = "This is a short document content."
        doc = Document(page_content=short_content, metadata={"page": 0})

        chunks = split_documents([doc])
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].page_content, short_content)
        self.assertEqual(DEFAULT_CHUNK_SIZE, 1000)
        self.assertEqual(DEFAULT_CHUNK_OVERLAP, 200)

    def test_split_documents_preserves_context_overlap(self) -> None:
        """Test that consecutive chunks retain overlapping context."""
        content = (
            "Sentence one describes the introduction. "
            "Sentence two elaborates on the system architecture. "
            "Sentence three discusses the data pipelines. "
            "Sentence four concludes the evaluation of results."
        )
        doc = Document(page_content=content, metadata={"page": 0})
        chunks = split_documents([doc], chunk_size=80, chunk_overlap=30)

        self.assertGreater(len(chunks), 1)
        # Check that there is common text between consecutive chunks
        first_chunk_end = chunks[0].page_content[-20:]
        self.assertTrue(
            any(first_chunk_end in chunk.page_content for chunk in chunks[1:]),
            "Context overlap was not found in subsequent chunk."
        )


if __name__ == "__main__":
    unittest.main()
