"""
FAISS vector store module.
Provides focused functions for creating, saving, and loading FAISS vector stores using LangChain.
"""

from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


def create_vector_store(
    documents: list[Document],
    embedding_model: Embeddings,
) -> FAISS:
    """Creates a FAISS vector store from a list of LangChain Document chunks.

    Args:
        documents: List of LangChain Document chunks to embed and index.
        embedding_model: Embedding model instance used to compute document embeddings.

    Returns:
        FAISS: An initialized FAISS vector store containing the documents.

    Raises:
        ValueError: If documents list is empty or embedding_model is None.
    """
    if not documents:
        raise ValueError("Cannot create a vector store from an empty list of documents.")
    if embedding_model is None:
        raise ValueError("An embedding model is required to create a vector store.")

    return FAISS.from_documents(documents=documents, embedding=embedding_model)


def save_vector_store(vector_store: FAISS, path: str | Path) -> None:
    """Persists a FAISS vector store to a local directory.

    Args:
        vector_store: The FAISS vector store instance to persist.
        path: Directory path where FAISS index files (index.faiss, index.pkl) will be saved.

    Raises:
        ValueError: If vector_store is None.
    """
    if vector_store is None:
        raise ValueError("Cannot save a null vector store instance.")

    target_path = Path(path)
    target_path.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(target_path))


def load_vector_store(path: str | Path, embedding_model: Embeddings) -> FAISS:
    """Loads an existing persisted FAISS vector store from a local directory.

    Args:
        path: Directory path containing the persisted FAISS index files.
        embedding_model: Embedding model instance compatible with the stored vector store.

    Returns:
        FAISS: Loaded FAISS vector store instance.

    Raises:
        FileNotFoundError: If the index directory or required index files do not exist.
        ValueError: If embedding_model is None.
    """
    if embedding_model is None:
        raise ValueError("An embedding model is required to load a FAISS vector store.")

    index_dir = Path(path)
    if not index_dir.exists() or not index_dir.is_dir():
        raise FileNotFoundError(f"FAISS index directory not found at: {path}")

    index_file = index_dir / "index.faiss"
    pkl_file = index_dir / "index.pkl"
    if not index_file.exists() or not pkl_file.exists():
        raise FileNotFoundError(
            f"FAISS index files (index.faiss, index.pkl) missing in directory: {path}"
        )

    return FAISS.load_local(
        str(index_dir),
        embeddings=embedding_model,
        allow_dangerous_deserialization=True,
    )
