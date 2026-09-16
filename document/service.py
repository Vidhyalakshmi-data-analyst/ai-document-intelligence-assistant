"""
Document processing application service.

Provides a thin application-facing orchestration function that coordinates
PDF loading, document chunking, embedding model creation, and FAISS
vector store creation.
"""

from pathlib import Path
from typing import Any

from embeddings.gemini_embeddings import create_embedding_model
from ingestion.pdf_loader import load_pdf
from ingestion.text_splitter import split_documents
from vectorstore.faiss_store import create_vector_store


def process_document(
    file_path: str | Path,
    api_key: str | None = None,
    embedding_model_name: str | None = None,
) -> Any:
    """Processes a PDF document and creates a queryable FAISS vector store.

    Orchestrates the ingestion, chunking, embedding, and vector store
    creation pipeline by delegating to existing specialized modules.

    Args:
        file_path: Path to the target PDF file.
        api_key: Optional Gemini API key. If not provided, resolved by the
            embedding module from application settings.
        embedding_model_name: Optional embedding model name. If not provided,
            resolved by the embedding module from application settings.

    Returns:
        FAISS: An initialized FAISS vector store containing the document chunks.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        ValueError: If the file path is invalid, API key is missing, or
            vector store creation fails.
    """
    documents = load_pdf(file_path)
    chunks = split_documents(documents)
    embedding_model = create_embedding_model(
        api_key=api_key,
        model_name=embedding_model_name,
    )
    return create_vector_store(
        documents=chunks,
        embedding_model=embedding_model,
    )
