"""
Google Gemini API service wrapper.

Encapsulates all direct interaction with the google-generativeai SDK,
providing text generation, structured JSON generation, and vision
(image + text) analysis, with retry logic and consistent error handling.
"""

from __future__ import annotations

import json
from typing import Any

import google.generativeai as genai
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.utils.config import get_settings
from src.utils.exceptions import GeminiAPIError
from src.utils.image_utils import EncodedImage
from src.utils.logger import get_logger

logger = get_logger(__name__)

_RETRYABLE_EXCEPTIONS = (Exception,)  # Gemini SDK raises generic exceptions on transient failures
_MAX_RETRY_ATTEMPTS = 3


class GeminiService:
    """
    Thin wrapper around the Google Gemini SDK for text and vision generation.

    This class is intentionally stateless per-call (no conversation memory
    here) — conversation context is passed in explicitly by callers so that
    memory management stays centralized in src/memory/.
    """

    def __init__(self) -> None:
        settings = get_settings()
        settings.validate_required_for_runtime()

        genai.configure(api_key=settings.gemini_api_key)

        self._text_model_name = settings.gemini_model_name
        self._vision_model_name = settings.gemini_vision_model_name

        self._text_model = genai.GenerativeModel(self._text_model_name)
        self._vision_model = genai.GenerativeModel(self._vision_model_name)

        logger.info(
            f"GeminiService initialized with text_model={self._text_model_name}, "
            f"vision_model={self._vision_model_name}"
        )

    @retry(
        stop=stop_after_attempt(_MAX_RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
        reraise=True,
    )
    def generate_text(
        self,
        prompt: str,
        *,
        system_instruction: str | None = None,
        temperature: float = 0.7,
        max_output_tokens: int = 1024,
    ) -> str:
        """
        Generate a plain-text response from Gemini.

        Args:
            prompt: The user/task prompt to send.
            system_instruction: Optional system-level instruction to steer behavior.
            temperature: Sampling temperature (0.0 - 1.0).
            max_output_tokens: Maximum tokens in the generated response.

        Returns:
            The generated text response.

        Raises:
            GeminiAPIError: If the API call fails after retries or returns no content.
        """
        try:
            model = self._text_model
            if system_instruction:
                model = genai.GenerativeModel(
                    self._text_model_name, system_instruction=system_instruction
                )

            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                ),
            )

            if not response.text:
                raise GeminiAPIError("Gemini returned an empty response.")

            return response.text.strip()

        except GeminiAPIError:
            raise
        except Exception as exc:
            logger.error(f"Gemini text generation failed: {exc}")
            raise GeminiAPIError("Failed to generate text response from Gemini.", details=str(exc)) from exc

    @retry(
        stop=stop_after_attempt(_MAX_RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
        reraise=True,
    )
    def generate_structured_json(
        self,
        prompt: str,
        *,
        system_instruction: str | None = None,
        temperature: float = 0.3,
    ) -> dict[str, Any]:
        """
        Generate a response and parse it as structured JSON.

        The prompt/system_instruction should explicitly instruct Gemini to
        return ONLY valid JSON with no markdown fences or preamble.

        Args:
            prompt: The task prompt requesting JSON output.
            system_instruction: Optional system instruction reinforcing JSON-only output.
            temperature: Lower temperature recommended for deterministic structured output.

        Returns:
            Parsed JSON as a Python dictionary.

        Raises:
            GeminiAPIError: If generation fails or the response is not valid JSON.
        """
        raw_text = self.generate_text(
            prompt,
            system_instruction=system_instruction,
            temperature=temperature,
        )

        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            # Strip markdown code fences (```json ... ``` or ``` ... ```)
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.error(f"Failed to parse Gemini JSON output: {cleaned[:500]}")
            raise GeminiAPIError(
                "Gemini did not return valid JSON.", details=str(exc)
            ) from exc

    @retry(
        stop=stop_after_attempt(_MAX_RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
        reraise=True,
    )
    def analyze_image(
        self,
        encoded_image: EncodedImage,
        prompt: str,
        *,
        temperature: float = 0.2,
    ) -> str:
        """
        Send an image + prompt to Gemini Vision and return the raw text response.

        Args:
            encoded_image: Base64-encoded image data prepared via image_utils.
            prompt: Instructions describing what to analyze/extract from the image.
            temperature: Sampling temperature; low by default for consistent extraction.

        Returns:
            The raw text response from Gemini Vision (often JSON-formatted per prompt).

        Raises:
            GeminiAPIError: If the vision API call fails or returns no content.
        """
        try:
            image_part = {
                "mime_type": encoded_image.mime_type,
                "data": encoded_image.base64_data,
            }

            response = self._vision_model.generate_content(
                [prompt, image_part],
                generation_config=genai.types.GenerationConfig(temperature=temperature),
            )

            if not response.text:
                raise GeminiAPIError("Gemini Vision returned an empty response.")

            return response.text.strip()

        except GeminiAPIError:
            raise
        except Exception as exc:
            logger.error(f"Gemini image analysis failed: {exc}")
            raise GeminiAPIError("Failed to analyze image with Gemini Vision.", details=str(exc)) from exc

    def analyze_image_structured(
        self,
        encoded_image: EncodedImage,
        prompt: str,
        *,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        """
        Analyze an image and parse the response as structured JSON.

        Args:
            encoded_image: Base64-encoded image data.
            prompt: Instructions requesting JSON-formatted extraction results.
            temperature: Sampling temperature for the vision model.

        Returns:
            Parsed JSON dictionary describing the image analysis result.

        Raises:
            GeminiAPIError: If the call fails or the response isn't valid JSON.
        """
        raw_text = self.analyze_image(encoded_image, prompt, temperature=temperature)

        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.error(f"Failed to parse Gemini Vision JSON output: {cleaned[:500]}")
            raise GeminiAPIError(
                "Gemini Vision did not return valid JSON.", details=str(exc)
            ) from exc


_service_instance: GeminiService | None = None


def get_gemini_service() -> GeminiService:
    """
    Return a lazily-initialized singleton GeminiService instance.

    Lazily initializing (rather than at import time) allows modules to
    import this file without immediately requiring a valid API key,
    which is useful for unit tests that mock this function.
    """
    global _service_instance
    if _service_instance is None:
        _service_instance = GeminiService()
    return _service_instance
