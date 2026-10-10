"""
Unit tests for Phase 5: Gemini Answer Generation.

All tests are offline -- no live Gemini API calls are made.
The ChatGoogleGenerativeAI model is mocked using unittest.mock.
"""

import unittest
from unittest.mock import MagicMock, patch

from config.settings import settings
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

    def setUp(self) -> None:
        """Ensure mock LLM mode is disabled so ChatGoogleGenerativeAI path is tested."""
        self.settings_patcher = patch.object(settings, "use_mock_llm", False)
        self.settings_patcher.start()
        self.addCleanup(self.settings_patcher.stop)

    def _make_mock_llm(self, content: object) -> MagicMock:
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

    @patch("llm.gemini_client.ChatGoogleGenerativeAI")
    def test_structured_text_blocks_with_extras_extracts_clean_text(
        self,
        mock_llm_cls: MagicMock,
    ) -> None:
        """Verify structured content blocks with extras/signatures extract only text."""
        structured_content = [
            {
                "type": "text",
                "text": "Based on the provided policy, employees are entitled to 25 days.",
                "extras": {"signature": "dummy-signature-token"},
            }
        ]
        mock_llm_cls.return_value = self._make_mock_llm(structured_content)
        result = generate_answer(
            question=_VALID_QUESTION,
            context=_VALID_CONTEXT,
            api_key=_VALID_API_KEY,
        )
        self.assertEqual(
            result,
            "Based on the provided policy, employees are entitled to 25 days.",
        )
        self.assertNotIn("extras", result)
        self.assertNotIn("{", result)

    @patch("llm.gemini_client.ChatGoogleGenerativeAI")
    def test_multiple_text_blocks_joined_in_original_order(
        self,
        mock_llm_cls: MagicMock,
    ) -> None:
        """Verify multiple text blocks are extracted and joined in order."""
        structured_content = [
            {"type": "text", "text": "Annual leave entitlement is 25 days."},
            {"type": "text", "text": "Carryover is permitted up to 5 days."},
        ]
        mock_llm_cls.return_value = self._make_mock_llm(structured_content)
        result = generate_answer(
            question=_VALID_QUESTION,
            context=_VALID_CONTEXT,
            api_key=_VALID_API_KEY,
        )
        self.assertEqual(
            result,
            "Annual leave entitlement is 25 days. Carryover is permitted up to 5 days.",
        )

    @patch("llm.gemini_client.ChatGoogleGenerativeAI")
    def test_unsupported_content_type_returns_fallback(
        self,
        mock_llm_cls: MagicMock,
    ) -> None:
        """Verify unsupported content types return application fallback response."""
        mock_llm_cls.return_value = self._make_mock_llm(12345)
        result = generate_answer(
            question=_VALID_QUESTION,
            context=_VALID_CONTEXT,
            api_key=_VALID_API_KEY,
        )
        self.assertIsInstance(result, str)
        self.assertIn("unable to generate", result.lower())

    @patch("llm.gemini_client.ChatGoogleGenerativeAI")
    def test_empty_list_content_returns_fallback(
        self,
        mock_llm_cls: MagicMock,
    ) -> None:
        """Verify empty content block list returns application fallback response."""
        mock_llm_cls.return_value = self._make_mock_llm([])
        result = generate_answer(
            question=_VALID_QUESTION,
            context=_VALID_CONTEXT,
            api_key=_VALID_API_KEY,
        )
        self.assertIsInstance(result, str)
        self.assertIn("unable to generate", result.lower())

    @patch("llm.gemini_client.ChatGoogleGenerativeAI")
    def test_non_text_dict_blocks_ignored_returns_fallback(
        self,
        mock_llm_cls: MagicMock,
    ) -> None:
        """Verify dictionary blocks without text string return fallback response."""
        mock_llm_cls.return_value = self._make_mock_llm([{"type": "image", "data": "dummy"}])
        result = generate_answer(
            question=_VALID_QUESTION,
            context=_VALID_CONTEXT,
            api_key=_VALID_API_KEY,
        )
        self.assertIsInstance(result, str)
        self.assertIn("unable to generate", result.lower())


class TestGenerateAnswerMockMode(unittest.TestCase):
    """Tests for generate_answer() when USE_MOCK_LLM is enabled."""

    @patch("llm.gemini_client.ChatGoogleGenerativeAI")
    @patch("llm.gemini_client.settings")
    def test_mock_mode_returns_deterministic_grounded_response(
        self,
        mock_settings: MagicMock,
        mock_chat_cls: MagicMock,
    ) -> None:
        """Verify mock mode produces a grounded response without calling ChatGoogleGenerativeAI."""
        mock_settings.use_mock_llm = True
        question = "What is the annual leave entitlement?"
        context = "Employees are entitled to 25 days of annual leave per year."

        answer = generate_answer(question=question, context=context)

        self.assertIn("25 days of annual leave", answer)
        mock_chat_cls.assert_not_called()

    @patch("llm.gemini_client.ChatGoogleGenerativeAI")
    @patch("llm.gemini_client.settings")
    def test_mock_mode_unanswerable_returns_fallback(
        self,
        mock_settings: MagicMock,
        mock_chat_cls: MagicMock,
    ) -> None:
        """Verify mock mode returns standard fallback when context lacks answer."""
        mock_settings.use_mock_llm = True
        question = "What is the policy for travel expenses?"
        context = "Employees are entitled to 25 days of annual leave per year."

        answer = generate_answer(question=question, context=context)

        self.assertEqual(
            answer,
            "The information is not available in the provided documents.",
        )
        mock_chat_cls.assert_not_called()

    @patch("llm.gemini_client.ChatGoogleGenerativeAI")
    @patch("llm.gemini_client.settings")
    def test_mock_mode_does_not_require_api_key(
        self,
        mock_settings: MagicMock,
        mock_chat_cls: MagicMock,
    ) -> None:
        """Verify mock mode does not raise ValueError for missing API key."""
        mock_settings.use_mock_llm = True
        mock_settings.gemini_api_key = None
        question = "What is the annual leave entitlement?"
        context = "Employees are entitled to 25 days of annual leave per year."

        answer = generate_answer(question=question, context=context, api_key=None)

        self.assertIn("25 days of annual leave", answer)
        mock_chat_cls.assert_not_called()

    @patch("llm.gemini_client.settings")
    def test_mock_mode_validates_empty_question(
        self,
        mock_settings: MagicMock,
    ) -> None:
        """Verify mock mode still raises ValueError on empty question."""
        mock_settings.use_mock_llm = True
        with self.assertRaises(ValueError) as ctx:
            generate_answer(question="   ", context=_VALID_CONTEXT)
        self.assertIn("Question must be a non-empty string", str(ctx.exception))

    @patch("llm.gemini_client.settings")
    def test_mock_mode_validates_empty_context(
        self,
        mock_settings: MagicMock,
    ) -> None:
        """Verify mock mode still raises ValueError on empty context."""
        mock_settings.use_mock_llm = True
        with self.assertRaises(ValueError) as ctx:
            generate_answer(question=_VALID_QUESTION, context="")
        self.assertIn("Context must be a non-empty string", str(ctx.exception))

    @patch("llm.gemini_client.ChatGoogleGenerativeAI")
    @patch("llm.gemini_client.settings")
    def test_mock_mode_does_not_return_generic_heading_when_entitlement_present(
        self,
        mock_settings: MagicMock,
        mock_chat_cls: MagicMock,
    ) -> None:
        """1. Generic 'Leave Policy' heading must not be returned when entitlement exists."""
        mock_settings.use_mock_llm = True
        context = (
            "Leave Policy\n"
            "Document No: HR-POL-001\n"
            "Sick Leave (SL): 8 days of leave with medical certificate."
        )

        answer = generate_answer(question="What is sick leave?", context=context)

        self.assertNotEqual(answer, "Leave Policy.")
        self.assertIn("8 days", answer)
        mock_chat_cls.assert_not_called()

    @patch("llm.gemini_client.ChatGoogleGenerativeAI")
    @patch("llm.gemini_client.settings")
    def test_mock_mode_returns_entitlement_for_adjacent_lines(
        self,
        mock_settings: MagicMock,
        mock_chat_cls: MagicMock,
    ) -> None:
        """2. Heading followed by entitlement on adjacent lines returns an answer containing '8 days'."""
        mock_settings.use_mock_llm = True
        context = (
            "Leave Policy\n"
            "Leave Entitlements\n"
            "Sick Leave (SL)\n"
            "8 days\n"
            "Medical certificate required for absence of 3 or more days."
        )

        answer = generate_answer(question="What is sick leave?", context=context)

        self.assertIn("8 days", answer)
        self.assertIn("Sick Leave", answer)
        self.assertNotEqual(answer, "Leave Policy.")
        mock_chat_cls.assert_not_called()

    @patch("llm.gemini_client.ChatGoogleGenerativeAI")
    @patch("llm.gemini_client.settings")
    def test_mock_mode_does_not_invent_entitlement_when_absent(
        self,
        mock_settings: MagicMock,
        mock_chat_cls: MagicMock,
    ) -> None:
        """3. Heuristic does not invent an entitlement when the context does not specify one."""
        mock_settings.use_mock_llm = True
        context = (
            "Leave Policy\n"
            "Sick leave requires approval from management."
        )

        answer = generate_answer(question="What is sick leave?", context=context)

        self.assertEqual(answer, "Sick leave requires approval from management.")
        self.assertNotIn("8 days", answer)
        mock_chat_cls.assert_not_called()

    @patch("llm.gemini_client.ChatGoogleGenerativeAI")
    @patch("llm.gemini_client.settings")
    def test_mock_mode_unanswerable_when_sick_leave_absent_from_policy(
        self,
        mock_settings: MagicMock,
        mock_chat_cls: MagicMock,
    ) -> None:
        """4. Returns fallback response when sick leave is absent, ignoring generic 'Leave Policy'."""
        mock_settings.use_mock_llm = True
        context = (
            "Leave Policy\n"
            "Annual leave is 20 days per year."
        )

        answer = generate_answer(question="What is sick leave?", context=context)

        self.assertEqual(
            answer,
            "The information is not available in the provided documents.",
        )
        mock_chat_cls.assert_not_called()

    @patch("llm.gemini_client.ChatGoogleGenerativeAI")
    @patch("llm.gemini_client.settings")
    def test_production_mode_uses_chat_google_generative_ai(
        self,
        mock_settings: MagicMock,
        mock_chat_cls: MagicMock,
    ) -> None:
        """Verify production mode (use_mock_llm=False) instantiates and calls ChatGoogleGenerativeAI."""
        mock_settings.use_mock_llm = False
        mock_settings.gemini_api_key = "test-api-key"
        mock_settings.gemini_generation_model = "gemini-3.7-flash"

        mock_response = MagicMock()
        mock_response.content = "Grounded production answer."
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_chat_cls.return_value = mock_llm

        answer = generate_answer(
            question=_VALID_QUESTION,
            context=_VALID_CONTEXT,
        )

        mock_chat_cls.assert_called_once_with(
            model="gemini-3.7-flash",
            google_api_key="test-api-key",
        )
        mock_llm.invoke.assert_called_once()
        self.assertEqual(answer, "Grounded production answer.")


if __name__ == "__main__":
    unittest.main()
