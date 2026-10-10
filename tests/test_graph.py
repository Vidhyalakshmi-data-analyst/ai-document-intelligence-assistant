import unittest
from unittest.mock import patch

from langchain_core.documents import Document

from graph.state import GraphState
from graph.nodes import (
    build_context_node,
    generate_answer_node,
    retrieve_node,
    extract_sources_node
)

from graph.workflow import build_graph



class TestGraphState(unittest.TestCase):
    """Tests for the LangGraph shared state."""

    def test_graph_state_can_be_created(self):
        state: GraphState = {
            "question": "What is the leave policy?",
            "retrieved_documents": [],
            "context": "",
            "answer": "",
            "sources": [],
        }

        self.assertEqual(state["question"], "What is the leave policy?")
        self.assertEqual(state["retrieved_documents"], [])
        self.assertEqual(state["context"], "")
        self.assertEqual(state["answer"], "")


class TestGraphNodes(unittest.TestCase):
    """Tests for LangGraph node functions."""

    def setUp(self):
        self.initial_state: GraphState = {
            "question": "What is the leave policy?",
            "retrieved_documents": [],
            "context": "",
            "answer": "",
            "sources": [],
        }

    @patch("graph.nodes.retrieve_documents")
    def test_retrieve_node(self, mock_retrieve):
        """Test that retrieve_node delegates to the retrieval function."""
        vector_store = object()

        documents = [
            Document(
                page_content="Employees receive 20 days of annual leave.",
                metadata={
    		"page": 1,
    		"source": "employee_policy.pdf",
		},
            )
        ]

        mock_retrieve.return_value = documents

        result = retrieve_node(
            state=self.initial_state,
            vector_store=vector_store,
        )

        mock_retrieve.assert_called_once_with(
            vector_store=vector_store,
            query="What is the leave policy?",
        )

        self.assertEqual(
            result["retrieved_documents"],
            documents,
        )

    def test_build_context_node(self):
        """Test that document contents are combined into context."""
        state: GraphState = {
            **self.initial_state,
            "retrieved_documents": [
                Document(
                    page_content="Annual leave is 20 days.",
                    metadata={
    			"page": 1,
    			"source": "employee_policy.pdf",
		},
                ),
                Document(
                    page_content="Leave requests require manager approval.",
                    metadata={
    			"page": 2,
    			"source": "employee_policy.pdf",
		},
                ),
            ],
        }

        result = build_context_node(state)

        self.assertEqual(
            result["context"],
            "Annual leave is 20 days.\n\n"
            "Leave requests require manager approval.",
        )

    def test_extract_sources_node(self):
        """Test that source metadata is extracted from retrieved documents."""
        state: GraphState = {
            **self.initial_state,
            "retrieved_documents": [
                Document(
                    page_content="Annual leave is 20 days.",
                    metadata={
                        "page": 1,
                        "source": "employee_policy.pdf",
                    },
                ),
                Document(
                    page_content="Leave requests require manager approval.",
                    metadata={
                        "page": 2,
                        "source": "employee_policy.pdf",
                    },
                ),
            ],
        }

        result = extract_sources_node(state)

        self.assertEqual(
            result["sources"],
            [
                {
                    "page": 1,
                    "source": "employee_policy.pdf",
                },
                {
                    "page": 2,
                    "source": "employee_policy.pdf",
                },
            ],
        )

    def test_extract_sources_node_deduplicates_same_page_chunks(self):
        """Test that multiple chunks from the same document and page are deduplicated."""
        state: GraphState = {
            **self.initial_state,
            "retrieved_documents": [
                Document(
                    page_content="First chunk from page 1.",
                    metadata={"page": 1, "source": "employee_policy.pdf"},
                ),
                Document(
                    page_content="Second chunk from page 1.",
                    metadata={"page": 1, "source": "employee_policy.pdf"},
                ),
                Document(
                    page_content="Third chunk from page 1.",
                    metadata={"page": 1, "source": "employee_policy.pdf"},
                ),
            ],
        }

        result = extract_sources_node(state)

        self.assertEqual(
            result["sources"],
            [
                {
                    "page": 1,
                    "source": "employee_policy.pdf",
                },
            ],
        )

    def test_extract_sources_node_preserves_different_pages(self):
        """Test that different pages from the same source document are not discarded."""
        state: GraphState = {
            **self.initial_state,
            "retrieved_documents": [
                Document(
                    page_content="Page 1 chunk.",
                    metadata={"page": 1, "source": "employee_policy.pdf"},
                ),
                Document(
                    page_content="Page 2 chunk.",
                    metadata={"page": 2, "source": "employee_policy.pdf"},
                ),
                Document(
                    page_content="Another page 1 chunk.",
                    metadata={"page": 1, "source": "employee_policy.pdf"},
                ),
            ],
        }

        result = extract_sources_node(state)

        self.assertEqual(
            result["sources"],
            [
                {
                    "page": 1,
                    "source": "employee_policy.pdf",
                },
                {
                    "page": 2,
                    "source": "employee_policy.pdf",
                },
            ],
        )

    def test_extract_sources_node_preserves_first_seen_ordering(self):
        """Test that source order reflects the first-seen retrieval order."""
        state: GraphState = {
            **self.initial_state,
            "retrieved_documents": [
                Document(
                    page_content="Doc B page 3.",
                    metadata={"page": 3, "source": "doc_b.pdf"},
                ),
                Document(
                    page_content="Doc A page 1.",
                    metadata={"page": 1, "source": "doc_a.pdf"},
                ),
                Document(
                    page_content="Doc B page 3 duplicate.",
                    metadata={"page": 3, "source": "doc_b.pdf"},
                ),
                Document(
                    page_content="Doc A page 2.",
                    metadata={"page": 2, "source": "doc_a.pdf"},
                ),
            ],
        }

        result = extract_sources_node(state)

        self.assertEqual(
            result["sources"],
            [
                {"page": 3, "source": "doc_b.pdf"},
                {"page": 1, "source": "doc_a.pdf"},
                {"page": 2, "source": "doc_a.pdf"},
            ],
        )


    @patch("graph.nodes.generate_answer")
    def test_generate_answer_node(self, mock_generate):
        """Test that generate_answer_node delegates to the Gemini client."""
        mock_generate.return_value = (
            "Employees receive 20 days of annual leave."
        )

        state: GraphState = {
            **self.initial_state,
            "context": "Employees receive 20 days of annual leave.",
        }

        result = generate_answer_node(
            state=state,
            api_key="test-key",
            model_name="test-model",
        )

        mock_generate.assert_called_once_with(
            question="What is the leave policy?",
            context="Employees receive 20 days of annual leave.",
            api_key="test-key",
            model_name="test-model",
        )

        self.assertEqual(
            result["answer"],
            "Employees receive 20 days of annual leave.",
        )

class TestGraphWorkflow(unittest.TestCase):
    """Tests for the compiled LangGraph workflow."""
    @patch("graph.nodes.generate_answer")
    @patch("graph.nodes.retrieve_documents")
    def test_graph_executes_in_expected_order(
        self,
        mock_retrieve,
        mock_generate,
    ):
        """Test that the complete graph executes successfully."""

        documents = [
            Document(
                page_content="Annual leave is 20 days.",
                metadata={
    		"page": 1,
    		"source": "employee_policy.pdf",
		},
            ),
            Document(
                page_content="Leave requests require manager approval.",
                metadata={
    			"page": 2,
    			"source": "employee_policy.pdf",
			},
            ),
        ]

        mock_retrieve.return_value = documents
        mock_generate.return_value = (
            "Employees receive 20 days of annual leave."
        )

        vector_store = object()

        graph = build_graph(
            vector_store=vector_store,
            api_key="test-key",
            model_name="test-model",
        )

        initial_state: GraphState = {
            "question": "What is the leave policy?",
            "retrieved_documents": [],
            "context": "",
            "answer": "",
            "sources": [],
        }

        result = graph.invoke(initial_state)

        mock_retrieve.assert_called_once_with(
            vector_store=vector_store,
            query="What is the leave policy?",
        )

        mock_generate.assert_called_once_with(
            question="What is the leave policy?",
            context=(
                "Annual leave is 20 days.\n\n"
                "Leave requests require manager approval."
            ),
            api_key="test-key",
            model_name="test-model",
        )

        self.assertEqual(
            result["question"],
            "What is the leave policy?",
        )
        self.assertEqual(
            result["retrieved_documents"],
            documents,
        )
        self.assertEqual(
            result["context"],
            "Annual leave is 20 days.\n\n"
            "Leave requests require manager approval.",
        )
        self.assertEqual(
            result["answer"],
            "Employees receive 20 days of annual leave.",
        )

        self.assertEqual(
            result["sources"],
            [
                {
                    "page": 1,
                    "source": "employee_policy.pdf",
                },
                {
                    "page": 2,
                    "source": "employee_policy.pdf",
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()