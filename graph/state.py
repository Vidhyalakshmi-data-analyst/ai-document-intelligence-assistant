"""
State definition for the LangGraph workflow.

This module is responsible solely for defining the state
shared between graph nodes.
"""

from typing import TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict):
    """State passed between nodes in the document intelligence workflow."""

    question: str
    retrieved_documents: list[Document]
    context: str
    answer: str