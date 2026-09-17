"""
AI Document Intelligence Assistant
Streamlit application entry point.

Provides a thin presentation layer over the document processing service
and chat service.
"""

from pathlib import Path
import tempfile
from typing import Any

import streamlit as st

from chat.service import answer_question
from config.settings import settings
from document.service import process_document


def init_session_state() -> None:
    """Initialize session state variables if not already set."""
    if "vector_store" not in st.session_state:
        st.session_state["vector_store"] = None
    if "processed_document_name" not in st.session_state:
        st.session_state["processed_document_name"] = None


def save_temporary_file(uploaded_file: Any) -> Path:
    """Save an uploaded file to a temporary file and return its Path.

    Args:
        uploaded_file: The Streamlit UploadedFile object.

    Returns:
        Path to the saved temporary file.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        return Path(tmp.name)


def process_uploaded_document(uploaded_file: Any) -> None:
    """Process an uploaded PDF file through the document processing service.

    Args:
        uploaded_file: The Streamlit UploadedFile object to process.
    """
    temp_path = save_temporary_file(uploaded_file)
    try:
        vector_store = process_document(file_path=temp_path)
        st.session_state["vector_store"] = vector_store
        st.session_state["processed_document_name"] = uploaded_file.name
    except ValueError as exc:
        st.session_state["vector_store"] = None
        st.session_state["processed_document_name"] = None
        if "API key" in str(exc):
            st.error("Gemini API key is not configured. Please check your configuration.")
        else:
            st.error(f"Failed to process document: {exc}")
    except Exception as exc:
        st.session_state["vector_store"] = None
        st.session_state["processed_document_name"] = None
        st.error(f"Failed to process document: {exc}")
    finally:
        if temp_path.exists():
            temp_path.unlink()


def display_sources(sources: list[dict[str, Any]]) -> None:
    """Render source attribution information in the UI.

    Args:
        sources: List of source dictionaries containing 'source' and 'page'.
    """
    if not sources:
        return

    st.subheader("Sources")
    for item in sources:
        source_name = item.get("source") or "Unknown source"
        page = item.get("page")
        page_label = f"Page {page}" if page is not None else "Page N/A"
        st.markdown(f"- **{source_name}** ({page_label})")


def handle_question(question: str) -> None:
    """Validate input, invoke the chat service, and display results.

    Args:
        question: The user query string.
    """
    vector_store = st.session_state.get("vector_store")
    if vector_store is None:
        st.warning("Please upload and process a PDF document first.")
        return

    cleaned_question = question.strip() if question else ""
    if not cleaned_question:
        st.warning("Please enter a question.")
        return

    try:
        response = answer_question(
            question=cleaned_question,
            vector_store=vector_store,
        )
    except ValueError as exc:
        if "API key" in str(exc):
            st.error("Gemini API key is not configured. Please check your configuration.")
        else:
            st.error(f"Failed to generate answer: {exc}")
        return
    except Exception as exc:
        st.error(f"Failed to generate answer: {exc}")
        return

    answer = response.get("answer") if isinstance(response, dict) else None
    if not answer or not str(answer).strip():
        st.error("The assistant was unable to produce an answer.")
        return

    st.subheader("Answer")
    st.write(str(answer).strip())
    sources = response.get("sources", []) if isinstance(response, dict) else []
    display_sources(sources)


def render_document_section() -> None:
    """Render the PDF upload and processing section of the UI."""
    st.header("1. Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload a PDF to process for document intelligence.",
    )

    if st.button("Process Document"):
        if uploaded_file is None:
            st.warning("Please select a PDF file to upload.")
        else:
            process_uploaded_document(uploaded_file)

    processed_doc = st.session_state.get("processed_document_name")
    if st.session_state.get("vector_store") is not None and processed_doc:
        st.success(f"Document ready: {processed_doc}")


def render_chat_section() -> None:
    """Render the question input and answer section of the UI."""
    st.header("2. Ask Questions")
    question = st.text_input(
        "Ask a question about the processed document:",
        placeholder="e.g., What is the annual leave entitlement?",
    )

    if st.button("Ask Question"):
        handle_question(question)


def render_ui() -> None:
    """Render the main user interface."""
    st.title(settings.app_name)
    st.caption("AI-powered document intelligence and grounded question answering.")
    render_document_section()
    st.divider()
    render_chat_section()


def main() -> None:
    """Streamlit application main entry point."""
    init_session_state()
    render_ui()


if __name__ == "__main__":
    main()
