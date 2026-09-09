"""
Unit and integration tests for Phase 3: Gemini Embeddings.
"""

import os
import unittest
from unittest.mock import patch

from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from embeddings.gemini_embeddings import create_embedding_model


class TestGeminiEmbeddings(unittest.TestCase):
    """Unit tests for the Gemini embedding model creation."""

    def test_create_embedding_model_success(self) -> None:
        """Test that embedding model is created with provided or configured settings."""
        model = create_embedding_model(
            api_key="fake-test-api-key",
            model_name="gemini-embedding-2-preview",
        )
        self.assertIsInstance(model, Embeddings)
        self.assertIsInstance(model, GoogleGenerativeAIEmbeddings)
        self.assertEqual(model.model, "gemini-embedding-2-preview")

    def test_create_embedding_model_default_model(self) -> None:
        """Test that default embedding model is gemini-embedding-2-preview when not specified."""
        model = create_embedding_model(api_key="fake-test-api-key")
        self.assertEqual(model.model, "gemini-embedding-2-preview")

    def test_create_embedding_model_missing_key_raises_error(self) -> None:
        """Test that missing Gemini API key raises a clear ValueError."""
        with patch("embeddings.gemini_embeddings.settings.gemini_api_key", None):
            with self.assertRaises(ValueError) as ctx:
                create_embedding_model(api_key=None)

            self.assertIn("Gemini API key is not configured", str(ctx.exception))

    def test_create_embedding_model_empty_key_raises_error(self) -> None:
        """Test that whitespace-only or empty Gemini API key raises a clear ValueError."""
        with patch("embeddings.gemini_embeddings.settings.gemini_api_key", "   "):
            with self.assertRaises(ValueError) as ctx:
                create_embedding_model(api_key="")

            self.assertIn("Gemini API key is not configured", str(ctx.exception))

    @unittest.skipUnless(
        os.getenv("GEMINI_API_KEY"),
        "Requires valid GEMINI_API_KEY environment variable for live API integration test.",
    )
    def test_embed_documents_live_api(self) -> None:
        """Live integration test: verifies embedding generation with actual Gemini API."""
        model = create_embedding_model()
        test_texts = [
            "Document intelligence enables automated text processing.",
            "FAISS provides efficient similarity search over vector embeddings.",
        ]
        embeddings = model.embed_documents(test_texts)

        self.assertEqual(len(embeddings), 2)
        self.assertTrue(all(isinstance(vec, list) and len(vec) > 0 for vec in embeddings))


if __name__ == "__main__":
    unittest.main()
