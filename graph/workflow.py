"""
LangGraph workflow definition for the document intelligence assistant.

This module is responsible solely for constructing and compiling
the document intelligence workflow.
"""

from typing import Any

from langgraph.graph import END, START, StateGraph

from graph.nodes import (
    build_context_node,
    generate_answer_node,
    retrieve_node,
)
from graph.state import GraphState


def build_graph(
    vector_store: Any,
    api_key: str | None = None,
    model_name: str | None = None,
):
    """Build and compile the document intelligence workflow."""

    def retrieve(state: GraphState) -> GraphState:
        """Run document retrieval using the supplied vector store."""
        return retrieve_node(
            state=state,
            vector_store=vector_store,
        )

    def generate_answer(state: GraphState) -> GraphState:
        """Generate an answer using the supplied Gemini configuration."""
        return generate_answer_node(
            state=state,
            api_key=api_key,
            model_name=model_name,
        )

    graph = StateGraph(GraphState)

    graph.add_node("retrieve", retrieve)
    graph.add_node("build_context", build_context_node)
    graph.add_node("generate_answer", generate_answer)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "build_context")
    graph.add_edge("build_context", "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph.compile()