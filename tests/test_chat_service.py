"""
Unit tests for the application service layer.
"""

import unittest
from unittest.mock import MagicMock, patch

from chat.service import answer_question


class TestAnswerQuestion(unittest.TestCase):
    """Tests for answer_question()."""

    @patch("chat.service.build_graph")
    def test_returns_answer_from_workflow(self, mock_build_graph):
        """Test that the service returns the answer produced by the workflow."""
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "question": "What is the leave policy?",
            "retrieved_documents": [],
            "context": "Annual leave is 20 days.",
            "answer": "Employees receive 20 days of annual leave.",
        }
        mock_build_graph.return_value = mock_graph

        result = answer_question(
            question="What is the leave policy?",
            vector_store=object(),
            api_key="test-key",
            model_name="test-model",
        )

        self.assertEqual(
    		result,
    		{
        		"answer": "Employees receive 20 days of annual leave.",
        		"sources": [],
    		},
	)

    @patch("chat.service.build_graph")
    def test_strips_question_before_workflow_invocation(
        self,
        mock_build_graph,
    ):
        """Test that surrounding whitespace is removed from the question."""
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
    		"answer": "Employees receive 20 days of annual leave.",
    		"sources": [
        		{
            		"page": 3,
            		"source": "employee_policy.pdf",
        		}
    			],
	}
        mock_build_graph.return_value = mock_graph

        vector_store = object()

        answer_question(
            question="  What is the leave policy?  ",
            vector_store=vector_store,
        )

        initial_state = mock_graph.invoke.call_args.args[0]

        self.assertEqual(
            initial_state["question"],
            "What is the leave policy?",
        )

    @patch("chat.service.build_graph")
    def test_builds_graph_with_supplied_configuration(
        self,
        mock_build_graph,
    ):
        """Test that the service passes workflow configuration correctly."""
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "answer": "Annual leave is 20 days.",
        }
        mock_build_graph.return_value = mock_graph

        vector_store = object()

        answer_question(
            question="What is the leave policy?",
            vector_store=vector_store,
            api_key="test-key",
            model_name="test-model",
        )

        mock_build_graph.assert_called_once_with(
            vector_store=vector_store,
            api_key="test-key",
            model_name="test-model",
        )

    @patch("chat.service.build_graph")
    def test_invokes_workflow_with_initial_state(
        self,
        mock_build_graph,
    ):
        """Test that the workflow receives the expected initial state."""
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "answer": "Annual leave is 20 days.",
        }
        mock_build_graph.return_value = mock_graph

        vector_store = object()

        answer_question(
            question="What is the leave policy?",
            vector_store=vector_store,
        )

        expected_state = {
            "question": "What is the leave policy?",
            "retrieved_documents": [],
            "context": "",
            "answer": "",
        }

        mock_graph.invoke.assert_called_once_with(expected_state)

    def test_empty_question_raises_value_error(self):
        """Test that an empty question is rejected."""
        with self.assertRaises(ValueError):
            answer_question(
                question="   ",
                vector_store=object(),
            )

    def test_non_string_question_raises_value_error(self):
        """Test that a non-string question is rejected."""
        with self.assertRaises(ValueError):
            answer_question(
                question=None,
                vector_store=object(),
            )

    def test_none_vector_store_raises_value_error(self):
        """Test that a missing vector store is rejected."""
        with self.assertRaises(ValueError):
            answer_question(
                question="What is the leave policy?",
                vector_store=None,
            )

    @patch("chat.service.build_graph")
    def test_empty_workflow_answer_raises_value_error(
        self,
        mock_build_graph,
    ):
        """Test that an empty workflow answer is rejected."""
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "answer": "",
        }
        mock_build_graph.return_value = mock_graph

        with self.assertRaises(ValueError):
            answer_question(
                question="What is the leave policy?",
                vector_store=object(),
            )

    @patch("chat.service.build_graph")
    def test_missing_workflow_answer_raises_value_error(
        self,
        mock_build_graph,
    ):
        """Test that a missing workflow answer is rejected."""
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {}
        mock_build_graph.return_value = mock_graph

        with self.assertRaises(ValueError):
            answer_question(
                question="What is the leave policy?",
                vector_store=object(),
            )


if __name__ == "__main__":
    unittest.main()
