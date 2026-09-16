"""
Unit tests for the document processing application service (Phase 10A).

Tests verify that process_document correctly orchestrates the existing
PDF loading, document chunking, embedding model creation, and FAISS
vector store creation modules without calling external APIs.
"""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from document.service import process_document


class TestProcessDocument(unittest.TestCase):
    """Tests for the process_document application service function."""

    def setUp(self) -> None:
        """Set up reusable test fixtures for mocked pipeline components."""
        self.test_file_path = "test_document.pdf"
        self.mock_documents = [
            Document(page_content="Policy content page 1", metadata={"page": 1}),
            Document(page_content="Policy content page 2", metadata={"page": 2}),
        ]
        self.mock_chunks = [
            Document(page_content="Policy chunk 1", metadata={"page": 1}),
            Document(page_content="Policy chunk 2", metadata={"page": 2}),
        ]
        self.mock_embedding_model = MagicMock()
        self.mock_vector_store = MagicMock()

    @patch("document.service.create_vector_store")
    @patch("document.service.create_embedding_model")
    @patch("document.service.split_documents")
    @patch("document.service.load_pdf")
    def test_process_document_success(
        self,
        mock_load_pdf: MagicMock,
        mock_split_documents: MagicMock,
        mock_create_embedding_model: MagicMock,
        mock_create_vector_store: MagicMock,
    ) -> None:
        """1. Successful document processing returns a vector store."""
        mock_load_pdf.return_value = self.mock_documents
        mock_split_documents.return_value = self.mock_chunks
        mock_create_embedding_model.return_value = self.mock_embedding_model
        mock_create_vector_store.return_value = self.mock_vector_store

        result = process_document(
            file_path=self.test_file_path,
            api_key="test-api-key",
            embedding_model_name="test-embedding-model",
        )

        self.assertEqual(result, self.mock_vector_store)

    @patch("document.service.create_vector_store")
    @patch("document.service.create_embedding_model")
    @patch("document.service.split_documents")
    @patch("document.service.load_pdf")
    def test_pdf_loader_called_with_supplied_path(
        self,
        mock_load_pdf: MagicMock,
        mock_split_documents: MagicMock,
        mock_create_embedding_model: MagicMock,
        mock_create_vector_store: MagicMock,
    ) -> None:
        """2. Existing PDF loader is called with the supplied file path."""
        mock_load_pdf.return_value = self.mock_documents
        mock_split_documents.return_value = self.mock_chunks
        mock_create_embedding_model.return_value = self.mock_embedding_model
        mock_create_vector_store.return_value = self.mock_vector_store

        file_path = Path("path/to/policy.pdf")
        process_document(file_path=file_path)

        mock_load_pdf.assert_called_once_with(file_path)

    @patch("document.service.create_vector_store")
    @patch("document.service.create_embedding_model")
    @patch("document.service.split_documents")
    @patch("document.service.load_pdf")
    def test_text_splitter_receives_loaded_documents(
        self,
        mock_load_pdf: MagicMock,
        mock_split_documents: MagicMock,
        mock_create_embedding_model: MagicMock,
        mock_create_vector_store: MagicMock,
    ) -> None:
        """3. Existing text splitter receives the loaded documents."""
        mock_load_pdf.return_value = self.mock_documents
        mock_split_documents.return_value = self.mock_chunks
        mock_create_embedding_model.return_value = self.mock_embedding_model
        mock_create_vector_store.return_value = self.mock_vector_store

        process_document(file_path=self.test_file_path)

        mock_split_documents.assert_called_once_with(self.mock_documents)

    @patch("document.service.create_vector_store")
    @patch("document.service.create_embedding_model")
    @patch("document.service.split_documents")
    @patch("document.service.load_pdf")
    def test_embedding_factory_called_with_configuration(
        self,
        mock_load_pdf: MagicMock,
        mock_split_documents: MagicMock,
        mock_create_embedding_model: MagicMock,
        mock_create_vector_store: MagicMock,
    ) -> None:
        """4. Existing embedding factory is called with the expected configuration."""
        mock_load_pdf.return_value = self.mock_documents
        mock_split_documents.return_value = self.mock_chunks
        mock_create_embedding_model.return_value = self.mock_embedding_model
        mock_create_vector_store.return_value = self.mock_vector_store

        process_document(
            file_path=self.test_file_path,
            api_key="custom-api-key",
            embedding_model_name="custom-embedding-model",
        )

        mock_create_embedding_model.assert_called_once_with(
            api_key="custom-api-key",
            model_name="custom-embedding-model",
        )

    @patch("document.service.create_vector_store")
    @patch("document.service.create_embedding_model")
    @patch("document.service.split_documents")
    @patch("document.service.load_pdf")
    def test_vector_store_receives_chunks_and_embedding_model(
        self,
        mock_load_pdf: MagicMock,
        mock_split_documents: MagicMock,
        mock_create_embedding_model: MagicMock,
        mock_create_vector_store: MagicMock,
    ) -> None:
        """5. Existing FAISS vector-store creation receives the split documents and embedding model."""
        mock_load_pdf.return_value = self.mock_documents
        mock_split_documents.return_value = self.mock_chunks
        mock_create_embedding_model.return_value = self.mock_embedding_model
        mock_create_vector_store.return_value = self.mock_vector_store

        process_document(file_path=self.test_file_path)

        mock_create_vector_store.assert_called_once_with(
            documents=self.mock_chunks,
            embedding_model=self.mock_embedding_model,
        )

    @patch("document.service.create_vector_store")
    @patch("document.service.create_embedding_model")
    @patch("document.service.split_documents")
    @patch("document.service.load_pdf")
    def test_vector_store_returned_unchanged(
        self,
        mock_load_pdf: MagicMock,
        mock_split_documents: MagicMock,
        mock_create_embedding_model: MagicMock,
        mock_create_vector_store: MagicMock,
    ) -> None:
        """6. The final vector store returned by the FAISS factory is returned unchanged."""
        mock_load_pdf.return_value = self.mock_documents
        mock_split_documents.return_value = self.mock_chunks
        mock_create_embedding_model.return_value = self.mock_embedding_model
        mock_create_vector_store.return_value = self.mock_vector_store

        result = process_document(file_path=self.test_file_path)

        self.assertIs(result, self.mock_vector_store)

    @patch("document.service.load_pdf")
    def test_loader_failure_propagated(
        self,
        mock_load_pdf: MagicMock,
    ) -> None:
        """7. A loader failure is propagated appropriately."""
        mock_load_pdf.side_effect = FileNotFoundError("PDF file not found at: missing.pdf")

        with self.assertRaises(FileNotFoundError) as ctx:
            process_document(file_path="missing.pdf")

        self.assertIn("PDF file not found", str(ctx.exception))

    @patch("document.service.split_documents")
    @patch("document.service.load_pdf")
    def test_splitter_failure_propagated(
        self,
        mock_load_pdf: MagicMock,
        mock_split_documents: MagicMock,
    ) -> None:
        """8. A splitter failure is propagated appropriately."""
        mock_load_pdf.return_value = self.mock_documents
        mock_split_documents.side_effect = ValueError("Text splitting failed")

        with self.assertRaises(ValueError) as ctx:
            process_document(file_path=self.test_file_path)

        self.assertIn("Text splitting failed", str(ctx.exception))

    @patch("document.service.create_embedding_model")
    @patch("document.service.split_documents")
    @patch("document.service.load_pdf")
    def test_embedding_creation_failure_propagated(
        self,
        mock_load_pdf: MagicMock,
        mock_split_documents: MagicMock,
        mock_create_embedding_model: MagicMock,
    ) -> None:
        """9. Embedding creation failure is propagated appropriately."""
        mock_load_pdf.return_value = self.mock_documents
        mock_split_documents.return_value = self.mock_chunks
        mock_create_embedding_model.side_effect = ValueError(
            "Gemini API key is not configured."
        )

        with self.assertRaises(ValueError) as ctx:
            process_document(file_path=self.test_file_path)

        self.assertIn("Gemini API key is not configured", str(ctx.exception))

    @patch("document.service.create_vector_store")
    @patch("document.service.create_embedding_model")
    @patch("document.service.split_documents")
    @patch("document.service.load_pdf")
    def test_vector_store_creation_failure_propagated(
        self,
        mock_load_pdf: MagicMock,
        mock_split_documents: MagicMock,
        mock_create_embedding_model: MagicMock,
        mock_create_vector_store: MagicMock,
    ) -> None:
        """10. Vector-store creation failure is propagated appropriately."""
        mock_load_pdf.return_value = self.mock_documents
        mock_split_documents.return_value = []
        mock_create_embedding_model.return_value = self.mock_embedding_model
        mock_create_vector_store.side_effect = ValueError(
            "Cannot create a vector store from an empty list of documents."
        )

        with self.assertRaises(ValueError) as ctx:
            process_document(file_path=self.test_file_path)

        self.assertIn("Cannot create a vector store from an empty list", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
