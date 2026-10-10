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


_NOT_AVAILABLE_RESPONSE = (
    "The information is not available in the provided documents."
)


_STOPWORDS: set[str] = {
    "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
    "is", "are", "was", "were", "be", "been", "being", "do", "does", "did",
    "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "about", "into", "through", "during", "before",
    "after", "above", "below", "can", "could", "should", "would", "may",
    "might", "must", "will", "shall", "it", "its", "they", "them", "their",
}


def _extract_query_terms(question: str) -> set[str]:
    """Extracts non-stopword query terms with punctuation stripped.

    Args:
        question: User query string.

    Returns:
        Set of normalized query terms.
    """
    return {
        token.strip("?.,!;:\"'()[]{}").lower()
        for token in question.split()
        if token.strip("?.,!;:\"'()[]{}").lower() not in _STOPWORDS
        and len(token.strip("?.,!;:\"'()[]{}")) > 1
    }


def _extract_context_candidates(context: str, query_terms: set[str]) -> list[str]:
    """Extracts sentence and multi-line heading candidates from context.

    Captures standard sentences as well as adjacent line pairs where a heading
    or category label matching query terms is followed by details or entitlement.

    Args:
        context: Retrieved context text.
        query_terms: Set of significant query terms.

    Returns:
        List of candidate text strings.
    """
    lines = [line.strip() for line in context.splitlines() if line.strip()]
    candidates: list[str] = []

    for line in lines:
        for sentence in line.split("."):
            s = sentence.strip()
            if s and len(s) > 2:
                candidates.append(s)

    for i in range(len(lines) - 1):
        curr = lines[i]
        curr_lower = curr.lower()
        if any(term in curr_lower for term in query_terms) and not curr.endswith((".", "!", "?")):
            nxt = lines[i + 1]
            candidates.append(f"{curr}: {nxt}")
            if i + 2 < len(lines):
                nxt2 = lines[i + 2]
                if not nxt.endswith((".", "!", "?")):
                    candidates.append(f"{curr}: {nxt}. {nxt2}")

    return candidates


def _score_candidate(
    candidate: str,
    query_terms: set[str],
    query_phrase: str,
) -> tuple[int, int, int, int]:
    """Computes relevance score tuple for a candidate against query terms.

    Criteria order:
    1. Count of distinct query terms matched.
    2. Exact multi-word query phrase occurrence.
    3. Topic prominence (starts with a query term).
    4. Informative details (contains digits/numbers).

    Args:
        candidate: Candidate sentence or combined line string.
        query_terms: Set of significant query terms.
        query_phrase: Cleaned multi-word query string.

    Returns:
        Score tuple for comparison.
    """
    c_lower = candidate.lower()
    matched_terms = {t for t in query_terms if t in c_lower}
    term_count = len(matched_terms)
    phrase_match = 1 if (query_phrase and query_phrase in c_lower) else 0
    starts_with_topic = 1 if any(c_lower.startswith(t) for t in query_terms) else 0
    has_digits = 1 if any(ch.isdigit() for ch in candidate) else 0

    return (term_count, phrase_match, starts_with_topic, has_digits)


def _generate_mock_answer(question: str, context: str) -> str:
    """Generates a deterministic grounded mock answer from context for development.

    Scores candidate sentences and heading-entitlement lines by query relevance
    to return specific information over generic titles, falling back when absent.

    Args:
        question: User query string.
        context: Retrieved document context string.

    Returns:
        Grounded answer string or unavailable fallback.
    """
    query_terms = _extract_query_terms(question)
    if not query_terms:
        return _NOT_AVAILABLE_RESPONSE

    candidates = _extract_context_candidates(context, query_terms)
    if not candidates:
        return _NOT_AVAILABLE_RESPONSE

    clean_phrase = " ".join(
        token.strip("?.,!;:\"'()[]{}").lower()
        for token in question.split()
        if token.strip("?.,!;:\"'()[]{}").lower() not in _STOPWORDS
    )

    min_required_matches = min(2, len(query_terms))
    scored: list[tuple[tuple[int, int, int, int], int, str]] = []

    for c in candidates:
        c_lower = c.lower()
        matched = {t for t in query_terms if t in c_lower}
        if len(matched) < min_required_matches:
            continue
        score = _score_candidate(c, query_terms, clean_phrase)
        scored.append((score, -len(c), c))

    if not scored:
        return _NOT_AVAILABLE_RESPONSE

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best = scored[0][2]
    return best if best.endswith((".", "!", "?")) else f"{best}."


def build_grounded_prompt(question: str, context: str) -> str:
    """Constructs the strict grounding prompt for the Gemini model.

    Args:
        question: The user question string.
        context: The retrieved document context to ground the answer in.

    Returns:
        Formatted prompt string ready to be sent to the LLM.
    """
    return _GROUNDING_TEMPLATE.format(context=context, question=question)


def _extract_text_content(content: object) -> str:
    """Extracts plain text from Gemini model response content.

    Handles plain strings as well as structured content block lists
    (e.g., [{'type': 'text', 'text': '...', 'extras': {...}}]).
    Ignores non-text metadata such as extras or signatures and joins
    multiple text blocks in their original order.

    Args:
        content: The raw response content from the Gemini model.

    Returns:
        The extracted answer text as a string, or an empty string if
        no valid text content could be extracted.
    """
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                cleaned = block.strip()
                if cleaned:
                    text_parts.append(cleaned)
            elif isinstance(block, dict):
                text_val = block.get("text")
                if isinstance(text_val, str):
                    cleaned = text_val.strip()
                    if cleaned:
                        text_parts.append(cleaned)
        return " ".join(text_parts).strip()

    return ""


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

    if getattr(settings, "use_mock_llm", False) is True:
        return _generate_mock_answer(question=question.strip(), context=context.strip())

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

    raw_content = getattr(response, "content", None)
    answer = _extract_text_content(raw_content)

    if not answer:
        return _FALLBACK_RESPONSE

    return answer
