"""Unit tests for src.services.gemini_service.GeminiService."""

from __future__ import annotations

import pytest

from src.utils.exceptions import GeminiAPIError


@pytest.fixture
def gemini_service(mocker):
    """Instantiate GeminiService with the underlying SDK configure/model calls mocked out."""
    mocker.patch("google.generativeai.configure")
    mock_model_cls = mocker.patch("google.generativeai.GenerativeModel")
    mock_model_cls.return_value = mocker.MagicMock()

    from src.services.gemini_service import GeminiService

    return GeminiService()


class TestGenerateText:
    def test_returns_stripped_text(self, gemini_service, mocker):
        mock_response = mocker.MagicMock(text="  Hello from Gemini  ")
        gemini_service._text_model.generate_content.return_value = mock_response

        result = gemini_service.generate_text("Say hello")
        assert result == "Hello from Gemini"

    def test_empty_response_raises_gemini_api_error(self, gemini_service, mocker):
        mock_response = mocker.MagicMock(text="")
        gemini_service._text_model.generate_content.return_value = mock_response

        with pytest.raises(GeminiAPIError):
            gemini_service.generate_text("Say hello")

    def test_sdk_exception_wrapped_as_gemini_api_error(self, gemini_service):
        gemini_service._text_model.generate_content.side_effect = RuntimeError("SDK exploded")

        with pytest.raises(GeminiAPIError):
            gemini_service.generate_text("Say hello")


class TestGenerateStructuredJson:
    def test_parses_clean_json(self, gemini_service, mocker):
        mocker.patch.object(gemini_service, "generate_text", return_value='{"key": "value"}')
        result = gemini_service.generate_structured_json("prompt")
        assert result == {"key": "value"}

    def test_strips_markdown_fences(self, gemini_service, mocker):
        mocker.patch.object(
            gemini_service, "generate_text", return_value='```json\n{"key": "value"}\n```'
        )
        result = gemini_service.generate_structured_json("prompt")
        assert result == {"key": "value"}

    def test_invalid_json_raises_gemini_api_error(self, gemini_service, mocker):
        mocker.patch.object(gemini_service, "generate_text", return_value="not valid json at all")
        with pytest.raises(GeminiAPIError):
            gemini_service.generate_structured_json("prompt")


class TestAnalyzeImage:
    def test_returns_text_response(self, gemini_service, mocker):
        from src.utils.image_utils import EncodedImage

        mock_response = mocker.MagicMock(text="A blue shirt.")
        gemini_service._vision_model.generate_content.return_value = mock_response

        encoded = EncodedImage(
            base64_data="fakebase64", mime_type="image/jpeg", original_size=(100, 100), resized_size=(100, 100)
        )
        result = gemini_service.analyze_image(encoded, "Describe this")
        assert result == "A blue shirt."

    def test_empty_vision_response_raises(self, gemini_service, mocker):
        from src.utils.image_utils import EncodedImage

        mock_response = mocker.MagicMock(text="")
        gemini_service._vision_model.generate_content.return_value = mock_response

        encoded = EncodedImage(
            base64_data="fakebase64", mime_type="image/jpeg", original_size=(100, 100), resized_size=(100, 100)
        )
        with pytest.raises(GeminiAPIError):
            gemini_service.analyze_image(encoded, "Describe this")
