"""
Semantic retrieval module.
Provides focused function to perform similarity search against a FAISS vector store.
"""

from typing import Any
from langchain_core.documents import Document


def retrieve_documents(
    vector_store: Any,
    query: str,
    top_k: int = 4,
) -> list[Document]:
    """Retrieves the top-K relevant document chunks for a query from a vector store.

    Args:
        vector_store: LangChain-compatible vector store (e.g., FAISS).
        query: User search query string.
        top_k: Number of relevant document chunks to return. Defaults to 4.

    Returns:
        List of relevant LangChain Document objects with preserved metadata.

    Raises:
        ValueError: If vector_store is None.
        ValueError: If query is not a non-empty string.
        ValueError: If top_k is not a positive integer.
    """
    if vector_store is None:
        raise ValueError("Vector store must not be None.")

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    return vector_store.similarity_search(query=query.strip(), k=top_k)
