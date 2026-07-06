"""
Input validation helpers for StyleSense AI.

Covers validation of uploaded images, budget values, and free-text user
input before they are passed to downstream tools or external APIs.
"""

from __future__ import annotations

import io
from typing import Final

from PIL import Image, UnidentifiedImageError

from src.utils.config import get_settings
from src.utils.exceptions import InvalidImageError, ValidationError
from src.utils.logger import get_logger

logger = get_logger(__name__)

_ALLOWED_IMAGE_FORMATS: Final[set[str]] = {"JPEG", "PNG", "WEBP"}
_MAX_MESSAGE_LENGTH: Final[int] = 2000


def validate_image_bytes(image_bytes: bytes) -> Image.Image:
    """
    Validate raw image bytes for size, format, and integrity.

    Args:
        image_bytes: Raw bytes of the uploaded image file.

    Returns:
        A PIL Image object opened from the validated bytes.

    Raises:
        InvalidImageError: If the image is too large, an unsupported
            format, empty, or corrupted.
    """
    settings = get_settings()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024

    if not image_bytes:
        raise InvalidImageError("Uploaded image is empty.")

    if len(image_bytes) > max_bytes:
        raise InvalidImageError(
            f"Image exceeds the maximum allowed size of {settings.max_upload_size_mb} MB.",
            details=f"Received {len(image_bytes) / (1024 * 1024):.2f} MB.",
        )

    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.verify()  # Validates integrity without fully decoding
        # Re-open because verify() leaves the file object unusable for further ops
        image = Image.open(io.BytesIO(image_bytes))
    except UnidentifiedImageError as exc:
        logger.warning("Uploaded file is not a recognizable image format.")
        raise InvalidImageError("Uploaded file is not a valid image.") from exc
    except Exception as exc:  # pragma: no cover - defensive catch-all
        logger.error(f"Unexpected error validating image: {exc}")
        raise InvalidImageError("Failed to validate uploaded image.") from exc

    if image.format not in _ALLOWED_IMAGE_FORMATS:
        raise InvalidImageError(
            f"Unsupported image format: {image.format}.",
            details=f"Allowed formats: {', '.join(_ALLOWED_IMAGE_FORMATS)}",
        )

    logger.debug(f"Image validated successfully: format={image.format}, size={image.size}")
    return image


def validate_budget(budget: float | int | None) -> float:
    """
    Validate a user-supplied budget value.

    Args:
        budget: The budget amount, expected to be a positive number.

    Returns:
        The validated budget as a float.

    Raises:
        ValidationError: If the budget is missing, negative, zero, or
            unreasonably large (> 1,000,000), which likely indicates
            input error.
    """
    if budget is None:
        raise ValidationError("Budget value is required.")

    try:
        budget_value = float(budget)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"Budget must be numeric, got: {budget!r}") from exc

    if budget_value <= 0:
        raise ValidationError("Budget must be a positive number greater than zero.")

    if budget_value > 1_000_000:
        raise ValidationError("Budget value seems unrealistically high. Please double-check.")

    return round(budget_value, 2)


def validate_user_message(message: str) -> str:
    """
    Validate and sanitize a free-text user chat message.

    Args:
        message: Raw user input string.

    Returns:
        The stripped, validated message string.

    Raises:
        ValidationError: If the message is empty or exceeds the max
            allowed length.
    """
    if message is None:
        raise ValidationError("Message cannot be None.")

    stripped = message.strip()

    if not stripped:
        raise ValidationError("Message cannot be empty.")

    if len(stripped) > _MAX_MESSAGE_LENGTH:
        raise ValidationError(
            f"Message exceeds maximum length of {_MAX_MESSAGE_LENGTH} characters.",
            details=f"Received {len(stripped)} characters.",
        )

    return stripped


def validate_occasion(occasion: str | None) -> str | None:
    """
    Lightweight normalization/validation for an 'occasion' field.

    Args:
        occasion: Free-text occasion description (e.g., "beach wedding").

    Returns:
        A cleaned, lowercase occasion string, or None if not provided.
    """
    if not occasion:
        return None

    cleaned = occasion.strip().lower()
    if len(cleaned) > 200:
        raise ValidationError("Occasion description is too long.")
    return cleaned
