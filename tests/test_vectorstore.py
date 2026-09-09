"""
Unit tests for Phase 3: FAISS Vector Store creation, persistence, and loading.
"""

import tempfile
import unittest
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings

from vectorstore.faiss_store import (
    create_vector_store,
    save_vector_store,
    load_vector_store,
)


class TestFaissVectorStore(unittest.TestCase):
    """Tests for FAISS vector store creation, persistence, loading, and metadata retention."""

    def setUp(self) -> None:
        """Set up offline fake embeddings and test documents."""
        self.embedding_model = FakeEmbeddings(size=768)
        self.test_docs = [
            Document(
                page_content="Introduction to AI Document Intelligence systems.",
                metadata={"source": "overview.pdf", "page": 0, "topic": "intro"},
            ),
            Document(
                page_content="Vector embeddings capture semantic relationships in high dimensions.",
                metadata={"source": "overview.pdf", "page": 1, "topic": "embeddings"},
            ),
            Document(
                page_content="FAISS facilitates fast nearest-neighbor lookups on vector indices.",
                metadata={"source": "overview.pdf", "page": 2, "topic": "indexing"},
            ),
        ]

    def test_create_vector_store_success(self) -> None:
        """Test that FAISS vector store is successfully created from document chunks."""
        vector_store = create_vector_store(self.test_docs, self.embedding_model)

        self.assertIsInstance(vector_store, FAISS)
        self.assertEqual(vector_store.index.ntotal, 3)

    def test_create_vector_store_empty_documents_raises_error(self) -> None:
        """Test that attempting to create a vector store from an empty list raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            create_vector_store([], self.embedding_model)

        self.assertIn("empty list of documents", str(ctx.exception))

    def test_create_vector_store_none_embeddings_raises_error(self) -> None:
        """Test that attempting to create a vector store without embeddings raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            create_vector_store(self.test_docs, None)  # type: ignore[arg-type]

        self.assertIn("embedding model is required", str(ctx.exception))

    def test_save_and_load_vector_store(self) -> None:
        """Test that a vector store can be saved locally and then loaded successfully."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_faiss_index"

            vector_store = create_vector_store(self.test_docs, self.embedding_model)
            save_vector_store(vector_store, save_path)

            # Check persisted index artifacts
            self.assertTrue((save_path / "index.faiss").exists())
            self.assertTrue((save_path / "index.pkl").exists())

            # Load the persisted vector store
            loaded_store = load_vector_store(save_path, self.embedding_model)

            self.assertIsInstance(loaded_store, FAISS)
            self.assertEqual(loaded_store.index.ntotal, 3)

    def test_document_metadata_preservation(self) -> None:
        """Test that document metadata is preserved after saving and reloading the vector store."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "metadata_test_index"

            vector_store = create_vector_store(self.test_docs, self.embedding_model)
            save_vector_store(vector_store, save_path)

            loaded_store = load_vector_store(save_path, self.embedding_model)

            # Verify stored documents in docstore retain their metadata
            docstore_docs = list(loaded_store.docstore._dict.values())
            self.assertEqual(len(docstore_docs), 3)

            topics = {doc.metadata.get("topic") for doc in docstore_docs}
            self.assertEqual(topics, {"intro", "embeddings", "indexing"})

            pages = {doc.metadata.get("page") for doc in docstore_docs}
            self.assertEqual(pages, {0, 1, 2})

            for doc in docstore_docs:
                self.assertEqual(doc.metadata.get("source"), "overview.pdf")

    def test_load_vector_store_missing_directory_raises_error(self) -> None:
        """Test that loading from a non-existent directory raises FileNotFoundError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            non_existent_path = Path(temp_dir) / "does_not_exist"

            with self.assertRaises(FileNotFoundError) as ctx:
                load_vector_store(non_existent_path, self.embedding_model)

            self.assertIn("FAISS index directory not found", str(ctx.exception))

    def test_load_vector_store_missing_index_files_raises_error(self) -> None:
        """Test that loading from an empty directory lacking index files raises FileNotFoundError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(FileNotFoundError) as ctx:
                load_vector_store(temp_dir, self.embedding_model)

            self.assertIn("FAISS index files (index.faiss, index.pkl) missing", str(ctx.exception))

    def test_save_vector_store_none_raises_error(self) -> None:
        """Test that saving a None vector store raises ValueError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError) as ctx:
                save_vector_store(None, temp_dir)  # type: ignore[arg-type]

            self.assertIn("Cannot save a null vector store instance", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
