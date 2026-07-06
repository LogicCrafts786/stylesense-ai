"""
Image analysis tool.

Wraps Gemini Vision analysis into a LangChain-compatible tool that the
shopping agent can invoke when a user uploads an image. Extracts garment
type, colors, style tags, and suitable occasions/seasons.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from src.prompts.image_analysis_prompts import GARMENT_ANALYSIS_PROMPT
from src.services.gemini_service import get_gemini_service
from src.utils.exceptions import GeminiAPIError, ImageAnalysisError, InvalidImageError
from src.utils.image_utils import EncodedImage, bytes_to_pil_image, encode_image_for_gemini
from src.utils.logger import get_logger
from src.utils.validators import validate_image_bytes

logger = get_logger(__name__)


def analyze_garment_image(image_bytes: bytes) -> dict[str, Any]:
    """
    Analyze an uploaded garment/outfit image and extract structured attributes.

    This is the core function (not a LangChain @tool wrapper) so it can be
    called directly from the LangGraph node with raw bytes, while a thin
    @tool wrapper below exposes it to any LangChain agent executor that
    prefers the tool-calling interface.

    Args:
        image_bytes: Raw bytes of the uploaded image file.

    Returns:
        A dictionary with keys: garment_type, dominant_colors,
        secondary_colors, style_tags, pattern, material_guess,
        fit_description, suitable_occasions, suitable_seasons, confidence.

    Raises:
        InvalidImageError: If the image fails validation.
        ImageAnalysisError: If encoding or analysis fails.
    """
    try:
        validated_image = validate_image_bytes(image_bytes)
        encoded: EncodedImage = encode_image_for_gemini(validated_image)

        gemini = get_gemini_service()
        result = gemini.analyze_image_structured(encoded, GARMENT_ANALYSIS_PROMPT)

        logger.info(
            f"Image analysis complete: garment_type={result.get('garment_type')}, "
            f"confidence={result.get('confidence')}"
        )
        return result

    except (InvalidImageError, GeminiAPIError):
        raise
    except Exception as exc:
        logger.error(f"Unexpected error during image analysis: {exc}")
        raise ImageAnalysisError("Failed to analyze the uploaded image.", details=str(exc)) from exc


@tool
def image_analysis_tool_fn(image_path: str) -> dict[str, Any]:
    """
    LangChain tool interface: analyze a garment image from a file path on disk
    and return structured attributes (garment type, colors, style, occasions).

    Args:
        image_path: Local filesystem path to the image file.

    Returns:
        Dictionary of extracted garment attributes.
    """
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    return analyze_garment_image(image_bytes)


def get_image_thumbnail_bytes(image_bytes: bytes) -> bytes:
    """
    Convenience helper: validate an image and return thumbnail-ready PIL data
    for the Streamlit UI to display alongside the chat message.

    Args:
        image_bytes: Raw uploaded image bytes.

    Returns:
        The original validated image as a PIL Image object (UI layer handles
        thumbnail rendering directly via st.image).
    """
    return bytes_to_pil_image(image_bytes)
