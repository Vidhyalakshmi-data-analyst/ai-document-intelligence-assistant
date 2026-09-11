"""
LangGraph node functions for the document intelligence workflow.

Each node is responsible for one step of the workflow and delegates
core functionality to the existing project modules.
"""

from typing import Any

from graph.state import GraphState
from llm.gemini_client import generate_answer
from retrieval.retriever import retrieve_documents


def retrieve_node(state: GraphState, vector_store: Any) -> GraphState:
    """Retrieve relevant documents for the user's question."""
    documents = retrieve_documents(
        vector_store=vector_store,
        query=state["question"],
    )

    return {
        **state,
        "retrieved_documents": documents,
    }


def build_context_node(state: GraphState) -> GraphState:
    """Combine retrieved document contents into a single context string."""
    context = "\n\n".join(
        document.page_content
        for document in state["retrieved_documents"]
    )

    return {
        **state,
        "context": context,
    }


def generate_answer_node(
    state: GraphState,
    api_key: str | None = None,
    model_name: str | None = None,
) -> GraphState:
    """Generate a grounded answer using the existing Gemini client."""
    answer = generate_answer(
        question=state["question"],
        context=state["context"],
        api_key=api_key,
        model_name=model_name,
    )

    return {
        **state,
        "answer": answer,
    }