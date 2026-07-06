"""
Weather service wrapper for occasion/weather-aware outfit reasoning.

Wraps the OpenWeatherMap API (or a compatible provider). Falls back
gracefully to a "context unknown" state if the API key is not configured,
since weather awareness is a nice-to-have enhancement, not a hard
dependency of the app.
"""

from __future__ import annotations

from dataclasses import dataclass

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.config import get_settings
from src.utils.exceptions import WeatherServiceError
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class WeatherContext:
    """
    Represents relevant weather conditions for outfit reasoning.

    Attributes:
        location: The queried location name.
        temperature_celsius: Current temperature in Celsius.
        condition: Short description (e.g., "light rain", "clear sky").
        season_hint: Derived season hint (e.g., "summer", "winter").
        is_available: False if weather data could not be retrieved.
    """

    location: str
    temperature_celsius: float | None
    condition: str | None
    season_hint: str | None
    is_available: bool = True


class WeatherService:
    """Client for retrieving current weather data to inform outfit recommendations."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._enabled = bool(self._settings.weather_api_key)
        if not self._enabled:
            logger.warning(
                "WEATHER_API_KEY not configured — weather-aware reasoning will be "
                "skipped, and the agent will rely solely on user-stated context."
            )

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        reraise=True,
    )
    def get_current_weather(self, location: str) -> WeatherContext:
        """
        Fetch current weather conditions for a given location.

        Args:
            location: City name or "City,CountryCode" (e.g., "Austin,US").

        Returns:
            A WeatherContext with temperature/condition data, or a
            not-available context if the service is disabled or fails.
        """
        if not self._enabled:
            return WeatherContext(
                location=location,
                temperature_celsius=None,
                condition=None,
                season_hint=None,
                is_available=False,
            )

        try:
            url = f"{self._settings.weather_api_base_url}/weather"
            params = {
                "q": location,
                "appid": self._settings.weather_api_key,
                "units": "metric",
            }
            response = requests.get(url, params=params, timeout=self._settings.scraper_timeout_seconds)
            response.raise_for_status()
            data = response.json()

            temperature = data.get("main", {}).get("temp")
            condition = (
                data.get("weather", [{}])[0].get("description")
                if data.get("weather")
                else None
            )

            return WeatherContext(
                location=location,
                temperature_celsius=temperature,
                condition=condition,
                season_hint=self._infer_season_hint(temperature),
                is_available=True,
            )

        except requests.RequestException as exc:
            logger.error(f"Weather API request failed for '{location}': {exc}")
            raise WeatherServiceError(
                f"Could not retrieve weather for '{location}'.", details=str(exc)
            ) from exc

    @staticmethod
    def _infer_season_hint(temperature_celsius: float | None) -> str | None:
        """
        Roughly categorize temperature into a season-like hint for outfit logic.

        Args:
            temperature_celsius: Current temperature reading, if available.

        Returns:
            One of "cold", "cool", "mild", "warm", "hot", or None if unknown.
        """
        if temperature_celsius is None:
            return None
        if temperature_celsius < 5:
            return "cold"
        if temperature_celsius < 15:
            return "cool"
        if temperature_celsius < 22:
            return "mild"
        if temperature_celsius < 30:
            return "warm"
        return "hot"


_service_instance: WeatherService | None = None


def get_weather_service() -> WeatherService:
    """Return a lazily-initialized singleton WeatherService instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = WeatherService()
    return _service_instance
