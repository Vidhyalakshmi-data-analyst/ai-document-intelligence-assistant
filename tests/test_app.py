"""
Unit tests for Streamlit UI error handling and boundary behavior (Phase 10C).

Verifies that expected runtime failures are caught at the UI boundary
and presented as user-friendly messages without exposing tracebacks or secrets,
and without corrupting session state.
"""

import unittest
from unittest.mock import MagicMock, patch

import app


class TestAppUIErrorHandling(unittest.TestCase):
    """Tests for UI error handling and boundary validation in app.py."""

    def setUp(self) -> None:
        """Initialize mock session state and mock uploaded file."""
        self.session_state: dict[str, object] = {
            "vector_store": None,
            "processed_document_name": None,
        }
        self.state_patcher = patch.object(app.st, "session_state", self.session_state)
        self.state_patcher.start()
        self.addCleanup(self.state_patcher.stop)

        self.mock_file = MagicMock()
        self.mock_file.name = "test_policy.pdf"
        self.mock_file.getvalue.return_value = b"%PDF-1.4 dummy content"

    @patch("app.save_temporary_file")
    @patch("app.process_document")
    @patch.object(app.st, "error")
    def test_process_uploaded_document_failure_displays_error(
        self,
        mock_error: MagicMock,
        mock_process_doc: MagicMock,
        mock_save_temp: MagicMock,
    ) -> None:
        """1. Processing failure is handled by the UI with a user-friendly error."""
        mock_save_temp.return_value.exists.return_value = False
        mock_process_doc.side_effect = ValueError("Corrupted PDF structure.")

        app.process_uploaded_document(self.mock_file)

        mock_error.assert_called_once()
        self.assertIn("Failed to process document", mock_error.call_args[0][0])
        self.assertIsNone(self.session_state["vector_store"])

    @patch("app.save_temporary_file")
    @patch("app.process_document")
    @patch.object(app.st, "error")
    def test_process_uploaded_document_missing_api_key_displays_config_error(
        self,
        mock_error: MagicMock,
        mock_process_doc: MagicMock,
        mock_save_temp: MagicMock,
    ) -> None:
        """Missing API key during processing displays a configuration error."""
        mock_save_temp.return_value.exists.return_value = False
        mock_process_doc.side_effect = ValueError("Gemini API key is not configured.")

        app.process_uploaded_document(self.mock_file)

        mock_error.assert_called_once()
        self.assertIn("Gemini API key is not configured", mock_error.call_args[0][0])
        self.assertIsNone(self.session_state["vector_store"])

    @patch("app.save_temporary_file")
    @patch("app.process_document")
    @patch.object(app.st, "error")
    def test_failed_processing_does_not_leave_successful_document_state(
        self,
        mock_error: MagicMock,
        mock_process_doc: MagicMock,
        mock_save_temp: MagicMock,
    ) -> None:
        """2. Failed processing does not leave a successful document state."""
        self.session_state["vector_store"] = MagicMock()
        self.session_state["processed_document_name"] = "previous.pdf"

        mock_save_temp.return_value.exists.return_value = False
        mock_process_doc.side_effect = RuntimeError("Extraction engine crashed.")

        app.process_uploaded_document(self.mock_file)

        mock_error.assert_called_once()
        self.assertIsNone(self.session_state["vector_store"])
        self.assertIsNone(self.session_state["processed_document_name"])

    @patch("app.answer_question")
    @patch.object(app.st, "error")
    def test_handle_question_service_failure_displays_error(
        self,
        mock_error: MagicMock,
        mock_answer_question: MagicMock,
    ) -> None:
        """3. Question-answering failure is handled by the UI."""
        self.session_state["vector_store"] = MagicMock()
        mock_answer_question.side_effect = RuntimeError("Model generation failed.")

        app.handle_question("What is the annual leave policy?")

        mock_error.assert_called_once()
        self.assertIn("Failed to generate answer", mock_error.call_args[0][0])

    @patch("app.answer_question")
    @patch.object(app.st, "error")
    def test_handle_question_missing_api_key_displays_config_error(
        self,
        mock_error: MagicMock,
        mock_answer_question: MagicMock,
    ) -> None:
        """Missing API key during question answering displays configuration error."""
        self.session_state["vector_store"] = MagicMock()
        mock_answer_question.side_effect = ValueError("Gemini API key is not configured.")

        app.handle_question("What is the annual leave policy?")

        mock_error.assert_called_once()
        self.assertIn("Gemini API key is not configured", mock_error.call_args[0][0])

    @patch("app.answer_question")
    @patch.object(app.st, "warning")
    def test_handle_question_no_vector_store_displays_warning(
        self,
        mock_warning: MagicMock,
        mock_answer_question: MagicMock,
    ) -> None:
        """4. Existing no-document behavior remains correct."""
        self.session_state["vector_store"] = None

        app.handle_question("What is the annual leave policy?")

        mock_warning.assert_called_once()
        self.assertIn("Please upload and process a PDF document first", mock_warning.call_args[0][0])
        mock_answer_question.assert_not_called()

    @patch("app.answer_question")
    @patch.object(app.st, "warning")
    def test_handle_question_empty_or_whitespace_displays_warning(
        self,
        mock_warning: MagicMock,
        mock_answer_question: MagicMock,
    ) -> None:
        """5. Existing empty-question behavior remains correct."""
        self.session_state["vector_store"] = MagicMock()

        app.handle_question("   \t \n  ")

        mock_warning.assert_called_once()
        self.assertIn("Please enter a question", mock_warning.call_args[0][0])
        mock_answer_question.assert_not_called()

    @patch("app.answer_question")
    @patch.object(app.st, "error")
    def test_handle_question_empty_answer_displays_error(
        self,
        mock_error: MagicMock,
        mock_answer_question: MagicMock,
    ) -> None:
        """Unexpected/empty response from chat service is handled safely."""
        self.session_state["vector_store"] = MagicMock()
        mock_answer_question.return_value = {"answer": "", "sources": []}

        app.handle_question("What is the annual leave policy?")

        mock_error.assert_called_once()
        self.assertIn("unable to produce an answer", mock_error.call_args[0][0])

    @patch("app.process_document")
    @patch("app.save_temporary_file")
    def test_process_uploaded_document_success_updates_session_state(
        self,
        mock_save_temp: MagicMock,
        mock_process_doc: MagicMock,
    ) -> None:
        """Successful document processing stores the vector store and document name."""
        vector_store = MagicMock()
        temp_path = MagicMock()
        temp_path.exists.return_value = False

        mock_save_temp.return_value = temp_path
        mock_process_doc.return_value = vector_store

        app.process_uploaded_document(self.mock_file)

        mock_save_temp.assert_called_once_with(self.mock_file)
        mock_process_doc.assert_called_once_with(file_path=temp_path)

        self.assertIs(
            self.session_state["vector_store"],
            vector_store,
        )
        self.assertEqual(
            self.session_state["processed_document_name"],
            "test_policy.pdf",
        )

    @patch("app.display_sources")
    @patch.object(app.st, "write")
    @patch.object(app.st, "subheader")
    @patch("app.answer_question")
    def test_handle_question_success_displays_answer_and_sources(
        self,
        mock_answer_question: MagicMock,
        mock_subheader: MagicMock,
        mock_write: MagicMock,
        mock_display_sources: MagicMock,
    ) -> None:
        """Successful question answering displays the answer and source information."""
        vector_store = MagicMock()
        sources = [
            {
                "page": 0,
                "source": "test_policy.pdf",
            }
        ]

        self.session_state["vector_store"] = vector_store

        mock_answer_question.return_value = {
            "answer": "Employees receive 20 days of annual leave.",
            "sources": sources,
        }

        app.handle_question("  How many days of annual leave?  ")

        mock_answer_question.assert_called_once_with(
            question="How many days of annual leave?",
            vector_store=vector_store,
        )

        mock_subheader.assert_called_once_with("Answer")
        mock_write.assert_called_once_with(
            "Employees receive 20 days of annual leave."
        )
        mock_display_sources.assert_called_once_with(sources)


if __name__ == "__main__":
    unittest.main()
