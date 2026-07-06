"""
Product comparison tool.

Uses Gemini to generate a structured side-by-side comparison of two or
more products, factoring in the user's stated priority (value, quality,
versatility, etc.) when provided.
"""

from __future__ import annotations

from langchain_core.tools import tool

from src.models.product import Product
from src.prompts.review_summary_prompts import build_product_comparison_prompt
from src.services.gemini_service import get_gemini_service
from src.tools.product_search_tool import get_products_by_ids
from src.utils.exceptions import GeminiAPIError, ValidationError
from src.utils.logger import get_logger

logger = get_logger(__name__)

_MIN_PRODUCTS_TO_COMPARE = 2


def compare_products(products: list[Product], user_priority: str | None = None) -> dict:
    """
    Generate a structured comparison of two or more products.

    Args:
        products: List of Product objects to compare (minimum 2).
        user_priority: Optional stated priority (e.g., "value for money").

    Returns:
        Dictionary with comparison_table, recommendation, and
        recommendation_reasoning.

    Raises:
        ValidationError: If fewer than 2 products are provided.
        GeminiAPIError: If the comparison generation fails.
    """
    if len(products) < _MIN_PRODUCTS_TO_COMPARE:
        raise ValidationError(
            f"At least {_MIN_PRODUCTS_TO_COMPARE} products are required for comparison.",
            details=f"Received {len(products)}.",
        )

    prompt = build_product_comparison_prompt(products, user_priority=user_priority)

    try:
        gemini = get_gemini_service()
        result = gemini.generate_structured_json(prompt, temperature=0.3)
        logger.info(f"Compared {len(products)} products; recommendation={result.get('recommendation')}")
        return result
    except GeminiAPIError as exc:
        logger.error(f"Product comparison generation failed: {exc}")
        raise


@tool
def comparison_tool_fn(product_ids: list[str], user_priority: str | None = None) -> dict:
    """
    LangChain tool interface: compare products by ID and return a
    structured comparison with a recommendation.

    Args:
        product_ids: List of product IDs to compare (looked up from catalog).
        user_priority: Optional priority the user cares about most.

    Returns:
        Structured comparison dictionary, or an error dict if comparison fails.
    """
    products = get_products_by_ids(product_ids)
    if len(products) < _MIN_PRODUCTS_TO_COMPARE:
        return {"error": "Could not find at least 2 valid products to compare."}

    try:
        return compare_products(products, user_priority=user_priority)
    except (ValidationError, GeminiAPIError) as exc:
        return {"error": str(exc)}
