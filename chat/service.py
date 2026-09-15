
"""
Application service for the document intelligence assistant.

This module provides a thin application-facing layer over the
existing LangGraph workflow.
"""

from typing import Any

from graph.state import GraphState
from graph.workflow import build_graph


def answer_question(
    question: str,
    vector_store: Any,
    api_key: str | None = None,
    model_name: str | None = None,
) -> str:
    """
    Execute the document intelligence workflow for a user question.

    The service validates the request, prepares the initial graph state,
    invokes the existing LangGraph workflow, and returns the final answer.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string.")

    if vector_store is None:
        raise ValueError("Vector store must not be None.")

    graph = build_graph(
        vector_store=vector_store,
        api_key=api_key,
        model_name=model_name,
    )

    initial_state: GraphState = {
        "question": question.strip(),
        "retrieved_documents": [],
        "context": "",
        "answer": "",
    }

    result = graph.invoke(initial_state)
    answer = result.get("answer")

    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Workflow returned an empty answer.")

    return {
          "answer": answer.strip(),
          "sources": result.get("sources", []),
    }

