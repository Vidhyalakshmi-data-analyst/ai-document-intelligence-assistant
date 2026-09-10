"""
Unit tests for Phase 5: Gemini Answer Generation.

All tests are offline -- no live Gemini API calls are made.
The ChatGoogleGenerativeAI model is mocked using unittest.mock.
"""

import unittest
from unittest.mock import MagicMock, patch

from llm.gemini_client import build_grounded_prompt, generate_answer


_VALID_QUESTION = "What is the annual leave entitlement?"
_VALID_CONTEXT = "Employees are entitled to 25 days of annual leave per year."
_VALID_API_KEY = "fake-test-api-key"
_MOCK_ANSWER = "Employees are entitled to 25 days of annual leave per year."


class TestBuildGroundedPrompt(unittest.TestCase):
    """Tests for the grounding prompt construction."""

    def test_grounding_instructions_present_in_prompt(self) -> None:
        """Test 9: Verify grounding instructions appear in the built prompt."""
        prompt = build_grounded_prompt(
            question=_VALID_QUESTION,
            context=_VALID_CONTEXT,
        )
        self.assertIsInstance(prompt, str)
        # Core grounding constraint keywords must be present
        self.assertIn("ONLY", prompt)
        self.assertIn("CONTEXT", prompt)
        self.assertIn("QUESTION", prompt)
        self.assertIn("not available in the provided documents", prompt)
        self.assertIn("Do NOT", prompt)
        # The actual question and context must be embedded in the prompt
        self.assertIn(_VALID_QUESTION, prompt)
        self.assertIn(_VALID_CONTEXT, prompt)


class TestGenerateAnswerValidation(unittest.TestCase):
    """Tests for input validation in generate_answer."""

    def test_empty_question_raises_value_error(self) -> None:
        """Test 1: Empty question raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            generate_answer(question="", context=_VALID_CONTEXT, api_key=_VALID_API_KEY)
        self.assertIn("Question must be a non-empty string", str(ctx.exception))

    def test_whitespace_question_raises_value_error(self) -> None:
        """Test 2: Whitespace-only question raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            generate_answer(question="   \t\n  ", context=_VALID_CONTEXT, api_key=_VALID_API_KEY)
        self.assertIn("Question must be a non-empty string", str(ctx.exception))

    def test_empty_context_raises_value_error(self) -> None:
        """Test 3: Empty context raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            generate_answer(question=_VALID_QUESTION, context="", api_key=_VALID_API_KEY)
        self.assertIn("Context must be a non-empty string", str(ctx.exception))

    def test_whitespace_context_raises_value_error(self) -> None:
        """Test 4: Whitespace-only context raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            generate_answer(question=_VALID_QUESTION, context="   \t  ", api_key=_VALID_API_KEY)
        self.assertIn("Context must be a non-empty string", str(ctx.exception))

    def test_missing_api_key_raises_value_error(self) -> None:
        """Test 5: Missing API key with no env fallback raises ValueError."""
        with patch("llm.gemini_client.settings") as mock_settings:
            mock_settings.gemini_api_key = None
            mock_settings.gemini_generation_model = "gemini-2.0-flash"
            with self.assertRaises(ValueError) as ctx:
                generate_answer(
                    question=_VALID_QUESTION,
                    context=_VALID_CONTEXT,
                    api_key=None,
                )
        self.assertIn("Gemini API key is not configured", str(ctx.exception))

    def test_empty_string_api_key_raises_value_error(self) -> None:
        """Test 5b: Empty-string API key with no env fallback raises ValueError."""
        with patch("llm.gemini_client.settings") as mock_settings:
            mock_settings.gemini_api_key = None
            mock_settings.gemini_generation_model = "gemini-2.0-flash"
            with self.assertRaises(ValueError) as ctx:
                generate_answer(
                    question=_VALID_QUESTION,
                    context=_VALID_CONTEXT,
                    api_key="",
                )
        self.assertIn("Gemini API key is not configured", str(ctx.exception))


class TestGenerateAnswerWithMockedModel(unittest.TestCase):
    """Tests for generate_answer behaviour with a mocked Gemini model."""

    def _make_mock_llm(self, content: str) -> MagicMock:
        """Helper: build a mock LLM whose invoke returns a response with .content."""
        mock_response = MagicMock()
        mock_response.content = content
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        return mock_llm

    @patch("llm.gemini_client.ChatGoogleGenerativeAI")
    def test_valid_inputs_return_mocked_answer(self, mock_llm_cls: MagicMock) -> None:
        """Test 6: Valid inputs with a mocked model return the expected answer."""
        mock_llm_cls.return_value = self._make_mock_llm(_MOCK_ANSWER)
        result = generate_answer(
            question=_VALID_QUESTION,
            context=_VALID_CONTEXT,
            api_key=_VALID_API_KEY,
            model_name="gemini-2.0-flash",
        )
        self.assertEqual(result, _MOCK_ANSWER)

    @patch("llm.gemini_client.ChatGoogleGenerativeAI")
    def test_return_type_is_str(self, mock_llm_cls: MagicMock) -> None:
        """Test 7: The return value is always a plain str."""
        mock_llm_cls.return_value = self._make_mock_llm(_MOCK_ANSWER)
        result = generate_answer(
            question=_VALID_QUESTION,
            context=_VALID_CONTEXT,
            api_key=_VALID_API_KEY,
        )
        self.assertIsInstance(result, str)

    @patch("llm.gemini_client.ChatGoogleGenerativeAI")
    def test_empty_model_response_returns_fallback(self, mock_llm_cls: MagicMock) -> None:
        """Test 8: An empty model response is handled safely with a fallback message."""
        mock_llm_cls.return_value = self._make_mock_llm("")
        result = generate_answer(
            question=_VALID_QUESTION,
            context=_VALID_CONTEXT,
            api_key=_VALID_API_KEY,
        )
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        self.assertIn("unable to generate", result.lower())

    @patch("llm.gemini_client.ChatGoogleGenerativeAI")
    def test_none_model_response_returns_fallback(self, mock_llm_cls: MagicMock) -> None:
        """Test 8b: A None content in model response is handled safely."""
        mock_response = MagicMock()
        mock_response.content = None
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_llm_cls.return_value = mock_llm
        result = generate_answer(
            question=_VALID_QUESTION,
            context=_VALID_CONTEXT,
            api_key=_VALID_API_KEY,
        )
        self.assertIsInstance(result, str)
        self.assertIn("unable to generate", result.lower())


if __name__ == "__main__":
    unittest.main()
