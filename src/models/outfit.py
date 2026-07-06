"""
Outfit data schema for StyleSense AI.

An Outfit represents a curated bundle of Product items assembled by the
bundle_generator_tool, along with the reasoning behind the pairing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.models.product import Product


@dataclass
class ColorHarmonyAnalysis:
    """
    Result of color-matching analysis performed on an outfit's items.

    Attributes:
        harmony_type: e.g., "complementary", "analogous", "monochromatic".
        score: A 0.0-1.0 confidence/quality score for the color pairing.
        explanation: Human-readable rationale for the color match.
    """

    harmony_type: str
    score: float
    explanation: str


@dataclass
class Outfit:
    """
    Represents a complete outfit recommendation.

    Attributes:
        outfit_id: Unique identifier for this outfit bundle.
        items: The list of Product objects making up the outfit.
        total_price: Sum of all item prices.
        occasion: The occasion this outfit was tailored for.
        weather_context: The weather/season context considered.
        style_summary: Short human-readable style description.
        reasoning: Detailed explanation of why these items were chosen together.
        color_harmony: Optional ColorHarmonyAnalysis for the outfit's palette.
        alternatives: Optional list of alternative Outfits (e.g., cheaper/pricier variants).
    """

    outfit_id: str
    items: list[Product]
    occasion: str | None = None
    weather_context: str | None = None
    style_summary: str = ""
    reasoning: str = ""
    color_harmony: ColorHarmonyAnalysis | None = None
    alternatives: list["Outfit"] = field(default_factory=list)

    @property
    def total_price(self) -> float:
        """Compute the total price of all items in this outfit."""
        return round(sum(item.price for item in self.items), 2)

    @property
    def item_count(self) -> int:
        """Number of items in this outfit."""
        return len(self.items)

    def is_within_budget(self, budget: float) -> bool:
        """Check whether this outfit's total price fits within a given budget."""
        return self.total_price <= budget

    def to_dict(self) -> dict:
        """Serialize the outfit to a plain dictionary for UI rendering or API responses."""
        return {
            "outfit_id": self.outfit_id,
            "items": [item.to_dict() for item in self.items],
            "total_price": self.total_price,
            "occasion": self.occasion,
            "weather_context": self.weather_context,
            "style_summary": self.style_summary,
            "reasoning": self.reasoning,
            "color_harmony": (
                {
                    "harmony_type": self.color_harmony.harmony_type,
                    "score": self.color_harmony.score,
                    "explanation": self.color_harmony.explanation,
                }
                if self.color_harmony
                else None
            ),
            "alternatives": [alt.to_dict() for alt in self.alternatives],
        }

    def __str__(self) -> str:  # pragma: no cover - trivial
        item_names = ", ".join(item.name for item in self.items)
        return f"Outfit[{self.outfit_id}]: {item_names} (${self.total_price:.2f})"
