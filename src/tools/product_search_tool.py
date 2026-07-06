"""
Product search tool.

Provides retrieval of candidate products from the catalog service and/or
vector store, based on structured filters extracted from user intent
(category, budget, colors, occasion, style).
"""

from __future__ import annotations

from langchain_core.tools import tool

from src.models.product import Product, ProductCategory
from src.services.product_catalog_service import get_product_catalog_service
from src.utils.exceptions import ProductCatalogError
from src.utils.logger import get_logger

logger = get_logger(__name__)


def search_products(
    *,
    categories: list[str] | None = None,
    max_price: float | None = None,
    colors: list[str] | None = None,
    style_tags: list[str] | None = None,
    occasion: str | None = None,
    limit: int = 20,
) -> list[Product]:
    """
    Search the product catalog for candidates matching the given filters.

    Args:
        categories: Product category strings (e.g., ["top", "shoes"]).
            Invalid/unknown category strings are silently skipped.
        max_price: Maximum acceptable price per item.
        colors: Preferred colors to match against product colors.
        style_tags: Preferred style keywords to match.
        occasion: Occasion to filter by (e.g., "wedding").
        limit: Maximum total number of products to return.

    Returns:
        A list of matching Product objects, capped at `limit`.
    """
    catalog = get_product_catalog_service()

    if not categories:
        results = catalog.filter_products(
            max_price=max_price,
            colors=colors,
            style_tags=style_tags,
            occasion=occasion,
        )
        return results[:limit]

    all_results: list[Product] = []
    for category_str in categories:
        try:
            category_enum = ProductCategory(category_str.lower())
        except ValueError:
            logger.warning(f"Unknown product category requested: '{category_str}' — skipping.")
            continue

        matches = catalog.filter_products(
            category=category_enum,
            max_price=max_price,
            colors=colors,
            style_tags=style_tags,
            occasion=occasion,
        )
        all_results.extend(matches)

    # De-duplicate while preserving order (in case of overlapping filters)
    seen_ids: set[str] = set()
    deduped: list[Product] = []
    for product in all_results:
        if product.product_id not in seen_ids:
            deduped.append(product)
            seen_ids.add(product.product_id)

    return deduped[:limit]


def get_products_by_ids(product_ids: list[str]) -> list[Product]:
    """
    Retrieve multiple products by their IDs, preserving input order.

    Args:
        product_ids: List of product IDs to fetch.

    Returns:
        List of found Product objects (missing IDs are silently skipped
        with a warning logged).
    """
    catalog = get_product_catalog_service()
    products: list[Product] = []
    for pid in product_ids:
        product = catalog.get_by_id(pid)
        if product is None:
            logger.warning(f"Product ID '{pid}' not found in catalog — skipping.")
            continue
        products.append(product)
    return products


@tool
def product_search_tool_fn(
    categories: list[str] | None = None,
    max_price: float | None = None,
    colors: list[str] | None = None,
    occasion: str | None = None,
) -> list[dict]:
    """
    LangChain tool interface: search the product catalog by category, price,
    color, and occasion filters. Returns a list of matching products as
    plain dictionaries.

    Args:
        categories: Product categories to search within.
        max_price: Maximum price filter.
        colors: Color filters.
        occasion: Occasion filter.

    Returns:
        List of matching products serialized as dictionaries.
    """
    try:
        results = search_products(
            categories=categories, max_price=max_price, colors=colors, occasion=occasion
        )
        return [p.to_dict() for p in results]
    except ProductCatalogError as exc:
        logger.error(f"Product search failed: {exc}")
        return []
