"""
Product catalog service.

Loads and queries the sample product catalog (data/sample_products.json).
This service acts as a stand-in for a real retailer/inventory API — the
interface is designed so it can be swapped for a live API client without
changing calling code in tools/agents.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from src.models.product import Product, ProductCategory
from src.utils.exceptions import ProductCatalogError
from src.utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_CATALOG_PATH = Path("data/sample_products.json")


class ProductCatalogService:
    """Provides read/query access to the product catalog dataset."""

    def __init__(self, catalog_path: Path | str = _DEFAULT_CATALOG_PATH) -> None:
        self._catalog_path = Path(catalog_path)
        self._products: list[Product] = []
        self._load_catalog()

    def _load_catalog(self) -> None:
        """
        Load and parse the product catalog JSON file into Product objects.

        Raises:
            ProductCatalogError: If the file is missing or contains invalid data.
        """
        if not self._catalog_path.exists():
            raise ProductCatalogError(
                f"Product catalog file not found at '{self._catalog_path}'.",
                details="Ensure data/sample_products.json exists in the repo.",
            )

        try:
            raw_data = json.loads(self._catalog_path.read_text(encoding="utf-8"))
            self._products = [Product.from_dict(item) for item in raw_data]
            logger.info(f"Loaded {len(self._products)} products from catalog.")
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.error(f"Failed to parse product catalog: {exc}")
            raise ProductCatalogError("Product catalog file is malformed.", details=str(exc)) from exc

    def get_all_products(self) -> list[Product]:
        """Return all products currently loaded in the catalog."""
        return list(self._products)

    def get_by_id(self, product_id: str) -> Product | None:
        """
        Retrieve a single product by its ID.

        Args:
            product_id: The unique product identifier.

        Returns:
            The matching Product, or None if not found.
        """
        return next((p for p in self._products if p.product_id == product_id), None)

    def filter_products(
        self,
        *,
        category: ProductCategory | None = None,
        max_price: float | None = None,
        colors: list[str] | None = None,
        style_tags: list[str] | None = None,
        occasion: str | None = None,
        in_stock_only: bool = True,
    ) -> list[Product]:
        """
        Filter the catalog by one or more criteria (all provided filters are ANDed).

        Args:
            category: Restrict to a specific ProductCategory.
            max_price: Maximum acceptable price.
            colors: Only include products with at least one matching color.
            style_tags: Only include products with at least one matching style tag.
            occasion: Only include products tagged for this occasion.
            in_stock_only: If True (default), exclude out-of-stock items.

        Returns:
            A list of Product objects matching all specified criteria.
        """
        results = self._products

        if in_stock_only:
            results = [p for p in results if p.in_stock]

        if category is not None:
            results = [p for p in results if p.category == category]

        if max_price is not None:
            results = [p for p in results if p.price <= max_price]

        if colors:
            wanted = {c.lower() for c in colors}
            results = [p for p in results if wanted.intersection({c.lower() for c in p.colors})]

        if style_tags:
            wanted_styles = {s.lower() for s in style_tags}
            results = [
                p for p in results if wanted_styles.intersection({s.lower() for s in p.style_tags})
            ]

        if occasion:
            occasion_lower = occasion.lower()
            results = [p for p in results if occasion_lower in {o.lower() for o in p.occasion_tags}]

        logger.debug(f"filter_products returned {len(results)} results.")
        return results

    def search_by_predicate(self, predicate: Callable[[Product], bool]) -> list[Product]:
        """
        Advanced filtering using an arbitrary predicate function.

        Args:
            predicate: A function taking a Product and returning True/False.

        Returns:
            All products for which the predicate returns True.
        """
        return [p for p in self._products if predicate(p)]


_service_instance: ProductCatalogService | None = None


def get_product_catalog_service() -> ProductCatalogService:
    """Return a lazily-initialized singleton ProductCatalogService instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = ProductCatalogService()
    return _service_instance
