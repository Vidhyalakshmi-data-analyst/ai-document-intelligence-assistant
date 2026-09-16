"""
Phase 9: Evaluation and Grounding Checks.

Deterministic offline evaluation tests verifying:
- Retrieval relevance for single and multiple topic documents.
- Grounding prompt constraints and instructions.
- Grounded answer generation for answerable questions.
- Application fallback behavior for unanswerable questions.
- Application fallback behavior for empty model responses.

No live Gemini API calls are made during these tests.
"""

import unittest
from unittest.mock import MagicMock, patch

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from llm.gemini_client import (
    _FALLBACK_RESPONSE,
    build_grounded_prompt,
    generate_answer,
)
from retrieval.retriever import retrieve_documents
from tests.test_retriever import DeterministicFakeEmbeddings


class TestRetrievalRelevance(unittest.TestCase):
    """Evaluation tests for retrieval relevance against controlled document sets."""

    def setUp(self) -> None:
        """Initialize deterministic embeddings and controlled document sets."""
        self.embeddings = DeterministicFakeEmbeddings()

    def test_single_document_retrieval_relevance(self) -> None:
        """Test Area 1: Verify relevant annual-leave document is retrieved among distinct topics."""
        documents = [
            Document(
                page_content="Employees receive 20 days of annual leave.",
                metadata={"source": "leave_policy.pdf", "page": 1},
            ),
            Document(
                page_content="Employees can work remotely two days per week.",
                metadata={"source": "remote_policy.pdf", "page": 1},
            ),
            Document(
                page_content="Employees must submit expense claims within 30 days.",
                metadata={"source": "expense_policy.pdf", "page": 1},
            ),
        ]
        vector_store = FAISS.from_documents(documents, self.embeddings)

        query = "What is the annual leave entitlement?"
        retrieved = retrieve_documents(vector_store, query=query, top_k=2)

        retrieved_texts = [doc.page_content for doc in retrieved]
        self.assertTrue(
            any("20 days of annual leave" in text for text in retrieved_texts),
            "Relevant annual leave document was not found in retrieved results.",
        )

    def test_multiple_relevant_documents_retrieval(self) -> None:
        """Test Area 6: Verify multiple relevant documents are retrieved when available."""
        documents = [
            Document(
                page_content="Employees receive 20 days of annual leave.",
                metadata={"source": "leave_policy.pdf", "page": 1},
            ),
            Document(
                page_content="Annual leave requests require manager approval.",
                metadata={"source": "leave_policy.pdf", "page": 2},
            ),
            Document(
                page_content="Employees can work remotely two days per week.",
                metadata={"source": "remote_policy.pdf", "page": 1},
            ),
            Document(
                page_content="Employees must submit expense claims within 30 days.",
                metadata={"source": "expense_policy.pdf", "page": 1},
            ),
        ]
        vector_store = FAISS.from_documents(documents, self.embeddings)

        query = "What are the annual leave rules?"
        retrieved = retrieve_documents(vector_store, query=query, top_k=2)

        retrieved_texts = [doc.page_content for doc in retrieved]
        self.assertEqual(len(retrieved), 2)
        self.assertTrue(
            any("20 days of annual leave" in text for text in retrieved_texts),
            "First relevant leave document missing from retrieved results.",
        )
        self.assertTrue(
            any("manager approval" in text for text in retrieved_texts),
            "Second relevant leave document missing from retrieved results.",
        )


class TestGroundingPrompt(unittest.TestCase):
    """Evaluation tests for strict grounding requirements in prompt construction."""

    def test_prompt_satisfies_all_grounding_requirements(self) -> None:
        """Test Area 2: Verify prompt enforces context, no outside knowledge, and fallback."""
        test_question = "What is the annual leave entitlement?"
        test_context = "Employees receive 20 days of annual leave."

        prompt = build_grounded_prompt(question=test_question, context=test_context)

        # 1. Includes the supplied context
        self.assertIn(test_context, prompt)

        # 2. Includes the supplied question
        self.assertIn(test_question, prompt)

        # 3. Instructs the model to use only the provided context
        self.assertIn("ONLY the context provided below", prompt)
        self.assertIn("Use exclusively the information given in the CONTEXT section", prompt)

        # 4. Instructs the model not to use outside knowledge
        self.assertIn("Do NOT use any outside knowledge or training data", prompt)

        # 5. Instructs the model not to invent facts
        self.assertIn("Do NOT invent, fabricate, or assume any facts", prompt)

        # 6. Contains the existing fallback response for information that is not available
        self.assertIn("The information is not available in the provided documents.", prompt)


class TestGroundedAnswerGeneration(unittest.TestCase):
    """Evaluation tests for grounded answer generation with mocked Gemini responses."""

    @patch("llm.gemini_client.ChatGoogleGenerativeAI")
    def test_answerable_question_returns_grounded_answer(
        self,
        mock_chat_cls: MagicMock,
    ) -> None:
        """Test Area 3: Verify answerable question produces expected grounded answer."""
        mock_response = MagicMock()
        mock_response.content = "Employees receive 20 days of annual leave."
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_chat_cls.return_value = mock_llm

        context = "Employees receive 20 days of annual leave."
        question = "How many days of annual leave do employees receive?"
        api_key = "test-api-key"
        model_name = "gemini-3.7-flash"

        answer = generate_answer(
            question=question,
            context=context,
            api_key=api_key,
            model_name=model_name,
        )

        self.assertEqual(answer, "Employees receive 20 days of annual leave.")
        mock_chat_cls.assert_called_once_with(
            model=model_name,
            google_api_key=api_key,
        )
        mock_llm.invoke.assert_called_once()

    @patch("llm.gemini_client.ChatGoogleGenerativeAI")
    def test_unanswerable_question_returns_fallback(
        self,
        mock_chat_cls: MagicMock,
    ) -> None:
        """Test Area 4: Verify unanswerable question returns application fallback response."""
        expected_fallback = "The information is not available in the provided documents."
        mock_response = MagicMock()
        mock_response.content = expected_fallback
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_chat_cls.return_value = mock_llm

        context = "Employees receive 20 days of annual leave."
        question = "What percentage bonus do employees receive?"
        api_key = "test-api-key"

        answer = generate_answer(
            question=question,
            context=context,
            api_key=api_key,
        )

        self.assertEqual(answer, expected_fallback)
        mock_llm.invoke.assert_called_once()

    @patch("llm.gemini_client.ChatGoogleGenerativeAI")
    def test_empty_model_response_returns_fallback_response(
        self,
        mock_chat_cls: MagicMock,
    ) -> None:
        """Test Area 5: Verify empty model response returns _FALLBACK_RESPONSE."""
        mock_response = MagicMock()
        mock_response.content = ""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_chat_cls.return_value = mock_llm

        context = "Employees receive 20 days of annual leave."
        question = "How many days of annual leave do employees receive?"
        api_key = "test-api-key"

        answer = generate_answer(
            question=question,
            context=context,
            api_key=api_key,
        )

        self.assertEqual(answer, _FALLBACK_RESPONSE)


if __name__ == "__main__":
    unittest.main()
