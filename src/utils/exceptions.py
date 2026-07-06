"""
Custom exception hierarchy for StyleSense AI.

Centralizing exceptions here allows calling code (agents, tools, UI) to
catch specific, meaningful errors instead of generic Exception instances,
and lets the UI layer present user-friendly messages while full details
are still logged.
"""

from __future__ import annotations


class StyleSenseBaseError(Exception):
    """Base exception for all StyleSense AI custom errors."""

    def __init__(self, message: str, *, details: str | None = None) -> None:
        self.message = message
        self.details = details
        super().__init__(self.message)

    def __str__(self) -> str:  # pragma: no cover - trivial
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class ConfigurationError(StyleSenseBaseError):
    """Raised when required configuration/environment variables are missing or invalid."""


class GeminiAPIError(StyleSenseBaseError):
    """Raised when a call to the Google Gemini API fails or returns an unexpected response."""


class ImageAnalysisError(StyleSenseBaseError):
    """Raised when image analysis (Gemini Vision) fails to process an uploaded image."""


class InvalidImageError(StyleSenseBaseError):
    """Raised when an uploaded image fails validation (size, format, corruption)."""


class ScraperError(StyleSenseBaseError):
    """Raised when web scraping (BeautifulSoup/Requests) fails or is blocked."""


class WeatherServiceError(StyleSenseBaseError):
    """Raised when the weather API call fails or returns invalid data."""


class ProductCatalogError(StyleSenseBaseError):
    """Raised when product catalog retrieval or parsing fails."""


class BudgetConstraintError(StyleSenseBaseError):
    """Raised when no products can be found within the user's stated budget."""


class VectorStoreError(StyleSenseBaseError):
    """Raised when vector store initialization, indexing, or querying fails."""


class MemoryError_(StyleSenseBaseError):
    """Raised when conversation or user-profile memory operations fail.

    Named MemoryError_ to avoid shadowing the built-in MemoryError.
    """


class AgentExecutionError(StyleSenseBaseError):
    """Raised when the LangGraph agent fails to execute or produce a valid final state."""


class ValidationError(StyleSenseBaseError):
    """Raised when user input fails validation rules."""
