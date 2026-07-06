"""
Image processing utilities built on Pillow.

Handles resizing, format conversion, and base64 encoding needed to send
images to the Gemini Vision API, as well as thumbnail generation for the
Streamlit UI.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass

from PIL import Image

from src.utils.exceptions import ImageAnalysisError
from src.utils.logger import get_logger

logger = get_logger(__name__)

_MAX_DIMENSION: int = 1024  # Max width/height sent to Gemini Vision to control payload size
_THUMBNAIL_SIZE: tuple[int, int] = (256, 256)


@dataclass(frozen=True)
class EncodedImage:
    """Container for a base64-encoded image ready for API transmission."""

    base64_data: str
    mime_type: str
    original_size: tuple[int, int]
    resized_size: tuple[int, int]


def resize_image_if_needed(image: Image.Image, max_dimension: int = _MAX_DIMENSION) -> Image.Image:
    """
    Downscale an image proportionally if either dimension exceeds max_dimension.

    Args:
        image: PIL Image to potentially resize.
        max_dimension: Maximum allowed width or height in pixels.

    Returns:
        The original image if within bounds, or a resized copy otherwise.
    """
    width, height = image.size
    if width <= max_dimension and height <= max_dimension:
        return image

    scale_factor = max_dimension / max(width, height)
    new_size = (int(width * scale_factor), int(height * scale_factor))
    resized = image.resize(new_size, Image.Resampling.LANCZOS)
    logger.debug(f"Resized image from {image.size} to {resized.size}")
    return resized


def encode_image_for_gemini(image: Image.Image) -> EncodedImage:
    """
    Convert a PIL Image into a base64-encoded JPEG suitable for the Gemini
    Vision API payload.

    Args:
        image: The validated PIL Image to encode.

    Returns:
        An EncodedImage dataclass with base64 data and metadata.

    Raises:
        ImageAnalysisError: If encoding fails for any reason.
    """
    try:
        original_size = image.size

        # Convert to RGB to ensure JPEG compatibility (handles RGBA/P modes)
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")

        resized = resize_image_if_needed(image)

        buffer = io.BytesIO()
        resized.save(buffer, format="JPEG", quality=88, optimize=True)
        encoded_bytes = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return EncodedImage(
            base64_data=encoded_bytes,
            mime_type="image/jpeg",
            original_size=original_size,
            resized_size=resized.size,
        )
    except Exception as exc:
        logger.error(f"Failed to encode image for Gemini: {exc}")
        raise ImageAnalysisError("Failed to prepare image for analysis.") from exc


def generate_thumbnail(image: Image.Image) -> Image.Image:
    """
    Generate a small thumbnail for display in the Streamlit chat UI.

    Args:
        image: The source PIL Image.

    Returns:
        A thumbnail-sized copy of the image (does not mutate the original).
    """
    thumbnail = image.copy()
    thumbnail.thumbnail(_THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
    return thumbnail


def bytes_to_pil_image(image_bytes: bytes) -> Image.Image:
    """
    Convert raw bytes to a PIL Image object.

    Args:
        image_bytes: Raw image file bytes.

    Returns:
        A PIL Image instance.

    Raises:
        ImageAnalysisError: If the bytes cannot be decoded as an image.
    """
    try:
        return Image.open(io.BytesIO(image_bytes))
    except Exception as exc:
        logger.error(f"Failed to convert bytes to PIL image: {exc}")
        raise ImageAnalysisError("Could not read uploaded image data.") from exc
