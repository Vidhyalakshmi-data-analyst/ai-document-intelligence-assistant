"""Application-level end-to-end integration tests."""

import unittest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from chat.service import answer_question
from document.service import process_document


class TestEndToEndPipeline(unittest.TestCase):
    """Test the complete application pipeline without external services."""

    @patch("document.service.create_vector_store")
    @patch("document.service.create_embedding_model")
    @patch("document.service.split_documents")
    @patch("document.service.load_pdf")
    @patch("graph.nodes.generate_answer")
    @patch("graph.nodes.retrieve_documents")
    def test_document_processing_to_answer_pipeline(
        self,
        mock_retrieve_documents,
        mock_generate_answer,
        mock_load_pdf,
        mock_split_documents,
        mock_create_embedding_model,
        mock_create_vector_store,
    ):
        """Process a document and answer a question through the real application pipeline."""
        loaded_documents = [
            Document(
                page_content="Employees receive 20 days of annual leave.",
                metadata={"page": 0, "source": "employee_handbook.pdf"},
            )
        ]

        split_documents = [
            Document(
                page_content="Employees receive 20 days of annual leave.",
                metadata={"page": 0, "source": "employee_handbook.pdf"},
            )
        ]

        vector_store = MagicMock(name="vector_store")
        embedding_model = MagicMock(name="embedding_model")

        mock_load_pdf.return_value = loaded_documents
        mock_split_documents.return_value = split_documents
        mock_create_embedding_model.return_value = embedding_model
        mock_create_vector_store.return_value = vector_store

        mock_retrieve_documents.return_value = split_documents

        mock_generate_answer.return_value = (
            "Employees receive 20 days of annual leave."
        )

        processed_vector_store = process_document(
            file_path="employee_handbook.pdf"
        )

        result = answer_question(
            question="How many days of annual leave do employees receive?",
            vector_store=processed_vector_store,
        )

        self.assertIs(processed_vector_store, vector_store)

        mock_create_vector_store.assert_called_once_with(
            documents=split_documents,
            embedding_model=embedding_model,
        )

        mock_retrieve_documents.assert_called_once_with(
            vector_store=vector_store,
            query="How many days of annual leave do employees receive?",
        )

        self.assertIsInstance(result, dict)
        self.assertTrue(result["answer"].strip())

        self.assertEqual(
            result["sources"],
            [
                {
                    "page": 0,
                    "source": "employee_handbook.pdf",
                }
            ],
        )

        self.assertEqual(
            result["answer"],
            "Employees receive 20 days of annual leave.",
        )


if __name__ == "__main__":
    unittest.main()