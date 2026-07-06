"""
Product data schema for StyleSense AI.

Defines the canonical representation of a shoppable item, used across
the product catalog service, vector store, and recommendation tools.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ProductCategory(str, Enum):
    """Enumeration of supported product categories."""

    TOP = "top"
    BOTTOM = "bottom"
    DRESS = "dress"
    OUTERWEAR = "outerwear"
    SHOES = "shoes"
    ACCESSORY = "accessory"
    BAG = "bag"
    JEWELRY = "jewelry"


@dataclass
class Product:
    """
    Represents a single product in the catalog.

    Attributes:
        product_id: Unique identifier for the product.
        name: Human-readable product name.
        category: The product's category (top, bottom, shoes, etc.).
        price: Price in the default currency (see Settings.default_currency).
        colors: List of dominant color names (e.g., ["navy", "white"]).
        style_tags: Free-form style descriptors (e.g., ["casual", "minimalist"]).
        brand: Brand or manufacturer name.
        description: Marketing/product description text.
        image_url: URL to a product image, if available.
        rating: Average customer rating (0.0 - 5.0), if known.
        review_count: Number of reviews backing the rating.
        in_stock: Whether the item is currently available.
        occasion_tags: Suitable occasions (e.g., ["wedding", "business"]).
        weather_tags: Suitable weather/climate conditions (e.g., ["summer", "rain"]).
        source_url: Original product page URL, used for scraping reviews.
    """

    product_id: str
    name: str
    category: ProductCategory
    price: float
    colors: list[str] = field(default_factory=list)
    style_tags: list[str] = field(default_factory=list)
    brand: str = "Unknown"
    description: str = ""
    image_url: str | None = None
    rating: float | None = None
    review_count: int = 0
    in_stock: bool = True
    occasion_tags: list[str] = field(default_factory=list)
    weather_tags: list[str] = field(default_factory=list)
    source_url: str | None = None

    def to_dict(self) -> dict:
        """Serialize the product to a plain dictionary (e.g., for JSON responses)."""
        return {
            "product_id": self.product_id,
            "name": self.name,
            "category": self.category.value,
            "price": self.price,
            "colors": self.colors,
            "style_tags": self.style_tags,
            "brand": self.brand,
            "description": self.description,
            "image_url": self.image_url,
            "rating": self.rating,
            "review_count": self.review_count,
            "in_stock": self.in_stock,
            "occasion_tags": self.occasion_tags,
            "weather_tags": self.weather_tags,
            "source_url": self.source_url,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Product":
        """Deserialize a Product from a plain dictionary (e.g., loaded from JSON)."""
        return cls(
            product_id=data["product_id"],
            name=data["name"],
            category=ProductCategory(data["category"]),
            price=float(data["price"]),
            colors=data.get("colors", []),
            style_tags=data.get("style_tags", []),
            brand=data.get("brand", "Unknown"),
            description=data.get("description", ""),
            image_url=data.get("image_url"),
            rating=data.get("rating"),
            review_count=data.get("review_count", 0),
            in_stock=data.get("in_stock", True),
            occasion_tags=data.get("occasion_tags", []),
            weather_tags=data.get("weather_tags", []),
            source_url=data.get("source_url"),
        )

    def matches_budget(self, max_budget: float) -> bool:
        """Return True if this product's price is within the given budget."""
        return self.price <= max_budget

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.brand} {self.name} (${self.price:.2f})"
