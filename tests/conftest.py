"""
Shared pytest fixtures for the StyleSense AI test suite.

Centralizes mock objects, sample data fixtures, and environment setup so
individual test modules stay focused on behavior rather than boilerplate.
No test in this suite should require a real GEMINI_API_KEY or network access.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.models.outfit import ColorHarmonyAnalysis, Outfit
from src.models.product import Product, ProductCategory


@pytest.fixture(autouse=True)
def _set_test_environment(monkeypatch, tmp_path):
    """
    Ensure required environment variables are present for every test,
    even though the actual value is a dummy key (no real API calls are
    made in unit tests due to mocking of service layer boundaries).
    """
    monkeypatch.setenv("GEMINI_API_KEY", "test-dummy-key")
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma_db"))

    # Clear cached settings singleton between tests to respect monkeypatched env vars.
    from src.utils.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def sample_product() -> Product:
    """A single representative Product for simple unit tests."""
    return Product(
        product_id="TEST-001",
        name="Test Oxford Shirt",
        category=ProductCategory.TOP,
        price=45.00,
        colors=["navy", "white"],
        style_tags=["classic", "business casual"],
        brand="TestBrand",
        description="A test product for unit tests.",
        rating=4.5,
        review_count=10,
        in_stock=True,
        occasion_tags=["office"],
        weather_tags=["mild"],
    )


@pytest.fixture
def sample_products(sample_product: Product) -> list[Product]:
    """A small list of varied Products for filtering/search tests."""
    second = Product(
        product_id="TEST-002",
        name="Test Chinos",
        category=ProductCategory.BOTTOM,
        price=55.00,
        colors=["khaki"],
        style_tags=["classic", "business casual"],
        brand="TestBrand",
        occasion_tags=["office"],
        weather_tags=["mild"],
    )
    third = Product(
        product_id="TEST-003",
        name="Test Sneakers",
        category=ProductCategory.SHOES,
        price=200.00,  # intentionally over typical test budgets
        colors=["white"],
        style_tags=["casual"],
        brand="TestBrand",
        occasion_tags=["casual outing"],
        weather_tags=["mild", "warm"],
    )
    return [sample_product, second, third]


@pytest.fixture
def sample_outfit(sample_product: Product) -> Outfit:
    """A representative Outfit object for UI/serialization tests."""
    return Outfit(
        outfit_id="outfit-test-001",
        items=[sample_product],
        occasion="office",
        weather_context="mild",
        style_summary="Clean business casual look",
        reasoning="The navy shirt suits an office setting.",
        color_harmony=ColorHarmonyAnalysis(
            harmony_type="neutral", score=0.9, explanation="Navy and white are classic neutrals."
        ),
    )


@pytest.fixture
def mock_gemini_text_response(mocker):
    """Factory fixture: patch GeminiService.generate_text to return a fixed string."""

    def _apply(return_value: str = "Mocked response text."):
        return mocker.patch(
            "src.services.gemini_service.GeminiService.generate_text", return_value=return_value
        )

    return _apply


@pytest.fixture
def mock_gemini_json_response(mocker):
    """Factory fixture: patch GeminiService.generate_structured_json to return a fixed dict."""

    def _apply(return_value: dict):
        return mocker.patch(
            "src.services.gemini_service.GeminiService.generate_structured_json",
            return_value=return_value,
        )

    return _apply
