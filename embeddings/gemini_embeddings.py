"""
Gemini embeddings module.
Responsible solely for configuring and creating the LangChain Gemini embedding model instance.
"""

from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config.settings import settings


def create_embedding_model(
    api_key: str | None = None,
    model_name: str | None = None,
) -> Embeddings:
    """Creates and returns a LangChain Google Gemini embedding model instance.

    Args:
        api_key: Optional Gemini API key. If not provided, loaded from application settings.
        model_name: Optional embedding model name. If not provided, loaded from application settings.

    Returns:
        Embeddings: Configured LangChain embedding model instance.

    Raises:
        ValueError: If no Gemini API key is configured or provided.
    """
    resolved_api_key = api_key or settings.gemini_api_key
    if not resolved_api_key or not resolved_api_key.strip():
        raise ValueError(
            "Gemini API key is not configured. Please set GEMINI_API_KEY in your "
            "environment or .env file, or provide it explicitly to create_embedding_model()."
        )

    resolved_model_name = model_name or settings.gemini_embedding_model

    return GoogleGenerativeAIEmbeddings(
        model=resolved_model_name,
        google_api_key=resolved_api_key,
    )
