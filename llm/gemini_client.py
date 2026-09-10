"""
Gemini LLM client module.
Responsible solely for generating grounded answers from a Gemini model
given a user question and retrieved document context.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from config.settings import settings


_FALLBACK_RESPONSE = (
    "I was unable to generate an answer based on the provided context."
)

_GROUNDING_TEMPLATE = (
    "You are a precise document assistant. Answer the user question using ONLY the "
    "context provided below.\n\n"
    "STRICT RULES:\n"
    "- Use exclusively the information given in the CONTEXT section.\n"
    "- Do NOT use any outside knowledge or training data.\n"
    "- Do NOT invent, fabricate, or assume any facts, policies, numbers, dates, "
    "names, or figures that are not explicitly stated in the context.\n"
    "- If the answer to the question is not present in the context, respond with: "
    "The information is not available in the provided documents.\n"
    "- Answer concisely and clearly.\n\n"
    "CONTEXT:\n"
    "{context}\n\n"
    "QUESTION:\n"
    "{question}\n\n"
    "ANSWER:"
)


def build_grounded_prompt(question: str, context: str) -> str:
    """Constructs the strict grounding prompt for the Gemini model.

    Args:
        question: The user question string.
        context: The retrieved document context to ground the answer in.

    Returns:
        Formatted prompt string ready to be sent to the LLM.
    """
    return _GROUNDING_TEMPLATE.format(context=context, question=question)


def generate_answer(
    question: str,
    context: str,
    api_key: str | None = None,
    model_name: str | None = None,
) -> str:
    """Generates a grounded answer to a question using only the supplied context.

    The model is strictly instructed not to use outside knowledge or fabricate
    information. If the answer is absent from the context, the response will
    clearly state that the information is not available in the provided documents.

    Args:
        question: The user question. Must be a non-empty string.
        context: Retrieved document context to ground the answer in.
                 Must be a non-empty string.
        api_key: Gemini API key. If not provided, resolved from application settings.
        model_name: Gemini generation model name. If not provided, resolved from
                    application settings (GEMINI_GENERATION_MODEL).

    Returns:
        The model generated answer as a plain string.

    Raises:
        ValueError: If question is empty or whitespace-only.
        ValueError: If context is empty or whitespace-only.
        ValueError: If no API key is configured or provided.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string.")

    if not isinstance(context, str) or not context.strip():
        raise ValueError("Context must be a non-empty string.")

    resolved_api_key = api_key or settings.gemini_api_key
    if not resolved_api_key or not resolved_api_key.strip():
        raise ValueError(
            "Gemini API key is not configured. Please set GEMINI_API_KEY in your "
            "environment or .env file, or provide it explicitly to generate_answer()."
        )

    resolved_model_name = model_name or settings.gemini_generation_model

    llm = ChatGoogleGenerativeAI(
        model=resolved_model_name,
        google_api_key=resolved_api_key,
    )

    prompt = build_grounded_prompt(question=question.strip(), context=context.strip())
    response = llm.invoke([HumanMessage(content=prompt)])

    answer = getattr(response, "content", None)

    if not answer or not str(answer).strip():
        return _FALLBACK_RESPONSE

    return str(answer).strip()
