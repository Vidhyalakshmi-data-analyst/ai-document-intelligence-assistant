"""
Document text splitting module.
Responsible solely for splitting LangChain Document objects into smaller chunks while preserving metadata.
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_CHUNK_SIZE: int = 1000
DEFAULT_CHUNK_OVERLAP: int = 200


def split_documents(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Splits a list of LangChain Documents into smaller chunks.

    Args:
        documents: List of LangChain Document objects to split.
        chunk_size: Maximum size of each text chunk in characters. Default is 1000.
        chunk_overlap: Overlap between consecutive chunks in characters. Default is 200.

    Returns:
        List of chunked LangChain Document objects with preserved metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)
