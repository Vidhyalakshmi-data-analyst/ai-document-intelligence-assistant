"""Document ingestion package."""

from ingestion.pdf_loader import load_pdf
from ingestion.text_splitter import split_documents

__all__ = ["load_pdf", "split_documents"]
