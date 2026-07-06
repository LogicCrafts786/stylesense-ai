"""Unit tests for src.utils.validators."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from src.utils.exceptions import InvalidImageError, ValidationError
from src.utils.validators import validate_budget, validate_image_bytes, validate_user_message


def _make_jpeg_bytes(size: tuple[int, int] = (100, 100)) -> bytes:
    """Helper: generate valid in-memory JPEG bytes for testing."""
    buffer = io.BytesIO()
    Image.new("RGB", size, color="blue").save(buffer, format="JPEG")
    return buffer.getvalue()


class TestValidateImageBytes:
    def test_valid_jpeg_passes(self):
        image_bytes = _make_jpeg_bytes()
        result = validate_image_bytes(image_bytes)
        assert result.format == "JPEG"

    def test_empty_bytes_raises(self):
        with pytest.raises(InvalidImageError):
            validate_image_bytes(b"")

    def test_corrupted_bytes_raises(self):
        with pytest.raises(InvalidImageError):
            validate_image_bytes(b"not a real image, just plain text bytes")

    def test_oversized_image_raises(self, monkeypatch):
        from src.utils import validators as validators_module

        monkeypatch.setattr(
            validators_module,
            "get_settings",
            lambda: type("S", (), {"max_upload_size_mb": 0.000001})(),
        )
        image_bytes = _make_jpeg_bytes((500, 500))
        with pytest.raises(InvalidImageError):
            validate_image_bytes(image_bytes)


class TestValidateBudget:
    @pytest.mark.parametrize("value,expected", [(50, 50.0), ("99.99", 99.99), (150.5, 150.5)])
    def test_valid_budgets(self, value, expected):
        assert validate_budget(value) == expected

    def test_none_raises(self):
        with pytest.raises(ValidationError):
            validate_budget(None)

    def test_zero_raises(self):
        with pytest.raises(ValidationError):
            validate_budget(0)

    def test_negative_raises(self):
        with pytest.raises(ValidationError):
            validate_budget(-10)

    def test_non_numeric_raises(self):
        with pytest.raises(ValidationError):
            validate_budget("not-a-number")

    def test_unreasonably_large_raises(self):
        with pytest.raises(ValidationError):
            validate_budget(5_000_000)


class TestValidateUserMessage:
    def test_valid_message_passes(self):
        assert validate_user_message("  Hello there  ") == "Hello there"

    def test_empty_message_raises(self):
        with pytest.raises(ValidationError):
            validate_user_message("   ")

    def test_none_raises(self):
        with pytest.raises(ValidationError):
            validate_user_message(None)  # type: ignore[arg-type]

    def test_overlong_message_raises(self):
        with pytest.raises(ValidationError):
            validate_user_message("x" * 3000)
