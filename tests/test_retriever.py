"""
Unit tests for Phase 4: Semantic Retrieval.
"""

import unittest
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from retrieval.retriever import retrieve_documents


class DeterministicFakeEmbeddings(Embeddings):
    """Deterministic lightweight embedding model for offline similarity testing."""

    def __init__(self, size: int = 128) -> None:
        self.size = size

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.size
        # Hash words to buckets to capture keyword overlap deterministically
        for token in text.lower().split():
            idx = abs(hash(token)) % self.size
            vec[idx] += 1.0
        # Normalize to unit length for cosine similarity
        norm = sum(x * x for x in vec) ** 0.5
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class TestRetriever(unittest.TestCase):
    """Tests for semantic retrieval functionality."""

    def setUp(self) -> None:
        """Create an in-memory FAISS vector store with test policy documents."""
        self.embeddings = DeterministicFakeEmbeddings()
        self.docs = [
            Document(
                page_content="Company Leave Policy: Full-time employees are entitled to 25 days of annual leave per year.",
                metadata={"source": "leave_policy.pdf", "page": 1, "department": "HR"},
            ),
            Document(
                page_content="Remote Work Policy: Eligible employees may work remotely up to two days per week with manager approval.",
                metadata={"source": "remote_work.pdf", "page": 2, "department": "Operations"},
            ),
            Document(
                page_content="Information Security Policy: Passwords must be at least 12 characters and updated every 90 days.",
                metadata={"source": "security_policy.pdf", "page": 1, "department": "IT"},
            ),
        ]
        self.vector_store = FAISS.from_documents(self.docs, self.embeddings)

    def test_successful_retrieval(self) -> None:
        """Test 1: Verify semantic retrieval returns relevant documents for a domain query."""
        query = "How many days of annual leave are employees entitled to?"
        results = retrieve_documents(self.vector_store, query, top_k=2)

        self.assertGreater(len(results), 0)
        # Most relevant document should be the leave policy
        self.assertIn("annual leave", results[0].page_content.lower())
        self.assertEqual(results[0].metadata.get("source"), "leave_policy.pdf")

    def test_top_k_behavior(self) -> None:
        """Test 2: Verify top_k controls the number of documents returned."""
        query = "policy rules and guidelines"

        results_k2 = retrieve_documents(self.vector_store, query, top_k=2)
        self.assertEqual(len(results_k2), 2)

        results_k1 = retrieve_documents(self.vector_store, query, top_k=1)
        self.assertEqual(len(results_k1), 1)

        # When top_k exceeds the number of available documents
        results_k10 = retrieve_documents(self.vector_store, query, top_k=10)
        self.assertEqual(len(results_k10), len(self.docs))

    def test_empty_or_whitespace_query_raises_error(self) -> None:
        """Test 3: Verify empty, whitespace-only, or None queries raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            retrieve_documents(self.vector_store, "")
        self.assertIn("Query must be a non-empty string", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            retrieve_documents(self.vector_store, "    \t \n  ")
        self.assertIn("Query must be a non-empty string", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            retrieve_documents(self.vector_store, None)  # type: ignore[arg-type]
        self.assertIn("Query must be a non-empty string", str(ctx.exception))

    def test_invalid_top_k_raises_error(self) -> None:
        """Test 4: Verify zero, negative, or non-integer top_k raises ValueError."""
        # Zero
        with self.assertRaises(ValueError) as ctx:
            retrieve_documents(self.vector_store, "valid query", top_k=0)
        self.assertIn("top_k must be a positive integer", str(ctx.exception))

        # Negative
        with self.assertRaises(ValueError) as ctx:
            retrieve_documents(self.vector_store, "valid query", top_k=-1)
        self.assertIn("top_k must be a positive integer", str(ctx.exception))

        # Float
        with self.assertRaises(ValueError) as ctx:
            retrieve_documents(self.vector_store, "valid query", top_k=2.5)  # type: ignore[arg-type]
        self.assertIn("top_k must be a positive integer", str(ctx.exception))

        # String
        with self.assertRaises(ValueError) as ctx:
            retrieve_documents(self.vector_store, "valid query", top_k="two")  # type: ignore[arg-type]
        self.assertIn("top_k must be a positive integer", str(ctx.exception))

        # Boolean (True evaluates as 1 in Python without explicit check)
        with self.assertRaises(ValueError) as ctx:
            retrieve_documents(self.vector_store, "valid query", top_k=True)  # type: ignore[arg-type]
        self.assertIn("top_k must be a positive integer", str(ctx.exception))

    def test_metadata_preservation(self) -> None:
        """Test 5: Verify retrieved documents retain all original metadata."""
        query = "annual leave entitlement"
        results = retrieve_documents(self.vector_store, query, top_k=1)

        self.assertEqual(len(results), 1)
        doc = results[0]
        self.assertEqual(doc.metadata.get("source"), "leave_policy.pdf")
        self.assertEqual(doc.metadata.get("page"), 1)
        self.assertEqual(doc.metadata.get("department"), "HR")

    def test_documents_remain_langchain_documents(self) -> None:
        """Test 6: Verify all returned items are LangChain Document instances."""
        query = "security requirements"
        results = retrieve_documents(self.vector_store, query, top_k=3)

        self.assertGreater(len(results), 0)
        for item in results:
            self.assertIsInstance(item, Document)
            self.assertIsInstance(item.page_content, str)
            self.assertIsInstance(item.metadata, dict)

    def test_none_vector_store_raises_error(self) -> None:
        """Test 7: Verify None vector store raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            retrieve_documents(None, "valid query", top_k=4)
        self.assertIn("Vector store must not be None", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
