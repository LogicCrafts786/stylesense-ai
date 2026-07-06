"""
Long-term user profile memory.

Persists user preferences (colors, styles, budget ranges, brands) across
conversation turns, and optionally across sessions if a persistent vector
store backend is configured. This is distinct from conversation_memory.py,
which only tracks the current session's raw chat transcript.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.models.user_preferences import UserPreferences
from src.utils.exceptions import MemoryError_
from src.utils.logger import get_logger

logger = get_logger(__name__)

_PROFILE_STORAGE_DIR = Path("data/user_profiles")


class UserProfileMemory:
    """
    Manages persistent UserPreferences objects, backed by simple JSON file
    storage for portability (no external DB dependency required to run
    the project out of the box).

    Note: For a production multi-user deployment, replace the file-based
    storage with a proper database (e.g., PostgreSQL, DynamoDB) behind the
    same interface.
    """

    def __init__(self, storage_dir: Path | str = _PROFILE_STORAGE_DIR) -> None:
        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, UserPreferences] = {}

    def _profile_path(self, user_id: str) -> Path:
        """Compute the JSON file path for a given user's profile."""
        safe_id = "".join(c for c in user_id if c.isalnum() or c in ("-", "_"))
        return self._storage_dir / f"{safe_id}.json"

    def get_or_create(self, user_id: str) -> UserPreferences:
        """
        Retrieve a user's preferences, loading from disk if not cached,
        or creating a fresh profile if none exists.

        Args:
            user_id: Unique identifier for the user/session.

        Returns:
            The UserPreferences object for this user.
        """
        if user_id in self._cache:
            return self._cache[user_id]

        profile_path = self._profile_path(user_id)
        if profile_path.exists():
            try:
                data = json.loads(profile_path.read_text(encoding="utf-8"))
                preferences = UserPreferences.from_dict(data)
                logger.debug(f"Loaded existing user profile for '{user_id}'.")
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning(f"Failed to load profile for '{user_id}', creating new: {exc}")
                preferences = UserPreferences(user_id=user_id)
        else:
            preferences = UserPreferences(user_id=user_id)
            logger.debug(f"Created new user profile for '{user_id}'.")

        self._cache[user_id] = preferences
        return preferences

    def save(self, preferences: UserPreferences) -> None:
        """
        Persist a UserPreferences object to disk and update the in-memory cache.

        Args:
            preferences: The UserPreferences object to save.

        Raises:
            MemoryError_: If writing to disk fails.
        """
        try:
            profile_path = self._profile_path(preferences.user_id)
            profile_path.write_text(json.dumps(preferences.to_dict(), indent=2), encoding="utf-8")
            self._cache[preferences.user_id] = preferences
            logger.debug(f"Saved user profile for '{preferences.user_id}'.")
        except OSError as exc:
            raise MemoryError_(
                f"Failed to save profile for user '{preferences.user_id}'.", details=str(exc)
            ) from exc

    def update_from_entities(
        self,
        user_id: str,
        *,
        colors_liked: list[str] | None = None,
        colors_disliked: list[str] | None = None,
        styles: list[str] | None = None,
        current_occasion: str | None = None,
        current_budget: float | None = None,
        current_weather_context: str | None = None,
    ) -> UserPreferences:
        """
        Update a user's profile based on entities extracted from their
        latest message, then persist the change.

        Args:
            user_id: The user/session identifier.
            colors_liked: Colors the user expressed positive sentiment toward.
            colors_disliked: Colors the user expressed negative sentiment toward.
            styles: New style preferences to merge in.
            current_occasion: Occasion stated in the current turn.
            current_budget: Budget stated in the current turn.
            current_weather_context: Weather/season context for the current turn.

        Returns:
            The updated UserPreferences object.
        """
        preferences = self.get_or_create(user_id)

        for color in colors_liked or []:
            preferences.update_color_preference(color, liked=True)
        for color in colors_disliked or []:
            preferences.update_color_preference(color, liked=False)

        for style in styles or []:
            style_lower = style.lower()
            if style_lower not in preferences.preferred_styles:
                preferences.preferred_styles.append(style_lower)

        if current_occasion:
            preferences.current_occasion = current_occasion
        if current_budget is not None:
            preferences.current_budget = current_budget
        if current_weather_context:
            preferences.current_weather_context = current_weather_context

        self.save(preferences)
        return preferences

    def clear_session_context(self, user_id: str) -> None:
        """
        Clear only the session-scoped fields (current_occasion, current_budget,
        current_weather_context) while preserving long-term style/color
        preferences. Useful when starting a new shopping "task" within the
        same user profile.

        Args:
            user_id: The user/session identifier.
        """
        preferences = self.get_or_create(user_id)
        preferences.current_occasion = None
        preferences.current_budget = None
        preferences.current_weather_context = None
        self.save(preferences)
        logger.debug(f"Cleared session context for user '{user_id}'.")


_memory_instance: UserProfileMemory | None = None


def get_user_profile_memory() -> UserProfileMemory:
    """Return a lazily-initialized singleton UserProfileMemory instance."""
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = UserProfileMemory()
    return _memory_instance
