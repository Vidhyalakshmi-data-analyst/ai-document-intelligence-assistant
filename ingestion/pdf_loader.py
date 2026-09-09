"""
PDF document loader module.
Responsible solely for loading and extracting content from PDF files into LangChain Document objects.
"""

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_pdf(file_path: str | Path) -> list[Document]:
    """Loads a PDF file and returns extracted LangChain Document objects with page metadata.

    Args:
        file_path: Path to the target PDF file.

    Returns:
        List of LangChain Document objects representing pages.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the path points to a directory rather than a file.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found at: {file_path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    loader = PyPDFLoader(str(path))
    return loader.load()
