"""
Streamlit image upload UI component.

Handles the file uploader widget, client-side preview, and validation
feedback before the image bytes are passed to the shopping agent.
"""

from __future__ import annotations

import streamlit as st

from src.utils.exceptions import InvalidImageError
from src.utils.logger import get_logger
from src.utils.validators import validate_image_bytes

logger = get_logger(__name__)

_SUPPORTED_TYPES = ["jpg", "jpeg", "png", "webp"]


def render_image_uploader() -> bytes | None:
    """
    Render a file uploader widget and return validated image bytes, or
    None if no file was uploaded or validation failed.

    Displays a thumbnail preview and inline error messages for invalid
    uploads (too large, wrong format, corrupted).

    Returns:
        Raw validated image bytes, or None.
    """
    uploaded_file = st.file_uploader(
        "📷 Upload a photo of an item or outfit (optional)",
        type=_SUPPORTED_TYPES,
        key="image_upload_widget",
        help="I'll analyze the colors, style, and category to help build recommendations.",
    )

    if uploaded_file is None:
        return None

    image_bytes = uploaded_file.getvalue()

    try:
        validated_image = validate_image_bytes(image_bytes)
        st.image(validated_image, caption="Uploaded image", width=200)
        return image_bytes
    except InvalidImageError as exc:
        st.error(f"⚠️ {exc.message}")
        logger.warning(f"Image upload rejected: {exc}")
        return None