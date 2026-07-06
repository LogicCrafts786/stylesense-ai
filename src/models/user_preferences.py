"""
User preference schema for StyleSense AI.

Captures both explicit, session-level preferences (stated budget, current
occasion) and longer-term style preferences that persist across sessions
via user_profile_memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class UserPreferences:
    """
    Represents a user's shopping preferences, combining session and
    long-term signal.

    Attributes:
        user_id: Unique identifier for the user/session.
        preferred_colors: Colors the user has expressed a liking for.
        disliked_colors: Colors the user wants to avoid.
        preferred_styles: Style descriptors the user favors (e.g., "minimalist").
        typical_budget_range: (min, max) tuple representing the user's usual budget.
        sizes: Mapping of category -> size (e.g., {"top": "M", "shoes": "9"}).
        favorite_brands: Brands the user has expressed preference for.
        current_occasion: The occasion relevant to the current conversation.
        current_budget: The budget stated for the current request, if any.
        current_weather_context: Weather/season relevant to the current request.
        last_updated: Timestamp of the last preference update.
    """

    user_id: str
    preferred_colors: list[str] = field(default_factory=list)
    disliked_colors: list[str] = field(default_factory=list)
    preferred_styles: list[str] = field(default_factory=list)
    typical_budget_range: tuple[float, float] | None = None
    sizes: dict[str, str] = field(default_factory=dict)
    favorite_brands: list[str] = field(default_factory=list)
    current_occasion: str | None = None
    current_budget: float | None = None
    current_weather_context: str | None = None
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def update_color_preference(self, color: str, liked: bool = True) -> None:
        """
        Record a new color preference signal.

        Args:
            color: The color name mentioned by the user.
            liked: True if the user expressed a positive sentiment, False if negative.
        """
        color_lower = color.lower()
        if liked:
            if color_lower not in self.preferred_colors:
                self.preferred_colors.append(color_lower)
            if color_lower in self.disliked_colors:
                self.disliked_colors.remove(color_lower)
        else:
            if color_lower not in self.disliked_colors:
                self.disliked_colors.append(color_lower)
            if color_lower in self.preferred_colors:
                self.preferred_colors.remove(color_lower)
        self.last_updated = datetime.utcnow()

    def to_dict(self) -> dict:
        """Serialize preferences to a plain dictionary for persistence."""
        return {
            "user_id": self.user_id,
            "preferred_colors": self.preferred_colors,
            "disliked_colors": self.disliked_colors,
            "preferred_styles": self.preferred_styles,
            "typical_budget_range": self.typical_budget_range,
            "sizes": self.sizes,
            "favorite_brands": self.favorite_brands,
            "current_occasion": self.current_occasion,
            "current_budget": self.current_budget,
            "current_weather_context": self.current_weather_context,
            "last_updated": self.last_updated.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UserPreferences":
        """Deserialize UserPreferences from a plain dictionary."""
        last_updated = data.get("last_updated")
        return cls(
            user_id=data["user_id"],
            preferred_colors=data.get("preferred_colors", []),
            disliked_colors=data.get("disliked_colors", []),
            preferred_styles=data.get("preferred_styles", []),
            typical_budget_range=tuple(data["typical_budget_range"])
            if data.get("typical_budget_range")
            else None,
            sizes=data.get("sizes", {}),
            favorite_brands=data.get("favorite_brands", []),
            current_occasion=data.get("current_occasion"),
            current_budget=data.get("current_budget"),
            current_weather_context=data.get("current_weather_context"),
            last_updated=datetime.fromisoformat(last_updated)
            if last_updated
            else datetime.utcnow(),
        )
