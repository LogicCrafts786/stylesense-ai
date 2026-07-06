"""
Weather and occasion reasoning tool.

Combines live weather data (when available) with occasion/season mapping
rules loaded from data/occasion_weather_map.json to recommend garment
weight/type suitability (e.g., "light layers", "waterproof outerwear").
"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.tools import tool

from src.services.weather_service import WeatherContext, get_weather_service
from src.utils.exceptions import WeatherServiceError
from src.utils.logger import get_logger

logger = get_logger(__name__)

_OCCASION_WEATHER_MAP_PATH = Path("data/occasion_weather_map.json")


def _load_occasion_weather_map() -> dict:
    """
    Load the occasion-to-weather-guidance mapping from the sample data file.

    Returns:
        A dictionary mapping occasion strings to guidance metadata. Returns
        an empty dict (with a warning logged) if the file is missing.
    """
    if not _OCCASION_WEATHER_MAP_PATH.exists():
        logger.warning(
            f"Occasion/weather map not found at '{_OCCASION_WEATHER_MAP_PATH}' — "
            "weather reasoning will rely on temperature only."
        )
        return {}
    return json.loads(_OCCASION_WEATHER_MAP_PATH.read_text(encoding="utf-8"))


def get_weather_context(location: str | None) -> WeatherContext | None:
    """
    Retrieve current weather context for a location, if provided.

    Args:
        location: City name, or None if the user didn't specify a location.

    Returns:
        A WeatherContext object, or None if no location was given.
    """
    if not location:
        return None

    try:
        weather_service = get_weather_service()
        return weather_service.get_current_weather(location)
    except WeatherServiceError as exc:
        logger.warning(f"Weather lookup failed for '{location}': {exc}")
        return None


def get_garment_guidance_for_context(
    *,
    occasion: str | None,
    season_hint: str | None,
) -> dict:
    """
    Combine occasion and season/weather hints into garment-type guidance
    (e.g., recommended layers, fabric weight, outerwear necessity).

    Args:
        occasion: The stated occasion (e.g., "wedding", "business meeting").
        season_hint: Season/temperature category (e.g., "cold", "warm", "hot").

    Returns:
        A dictionary with guidance keys: formality_level, recommended_layers,
        fabric_weight, outerwear_needed.
    """
    occasion_map = _load_occasion_weather_map()
    occasion_key = (occasion or "").lower().strip()
    occasion_data = occasion_map.get(occasion_key, {})

    formality_level = occasion_data.get("formality_level", "casual")

    fabric_weight_by_season = {
        "cold": "heavy",
        "cool": "medium-heavy",
        "mild": "medium",
        "warm": "light-medium",
        "hot": "light",
    }
    fabric_weight = fabric_weight_by_season.get(season_hint or "mild", "medium")

    outerwear_needed = season_hint in {"cold", "cool"}

    layers_by_season = {
        "cold": "3+ layers (base layer, sweater, heavy coat)",
        "cool": "2 layers (shirt/sweater + light jacket)",
        "mild": "1-2 layers",
        "warm": "1 light layer",
        "hot": "single breathable layer",
    }
    recommended_layers = layers_by_season.get(season_hint or "mild", "1-2 layers")

    return {
        "formality_level": formality_level,
        "recommended_layers": recommended_layers,
        "fabric_weight": fabric_weight,
        "outerwear_needed": outerwear_needed,
    }


@tool
def weather_tool_fn(location: str, occasion: str | None = None) -> dict:
    """
    LangChain tool interface: get current weather and derived garment
    guidance for a location and (optional) occasion.

    Args:
        location: City name to check weather for.
        occasion: Optional occasion to tailor formality/layer guidance.

    Returns:
        Dictionary combining weather data and garment guidance.
    """
    context = get_weather_context(location)
    season_hint = context.season_hint if context else None
    guidance = get_garment_guidance_for_context(occasion=occasion, season_hint=season_hint)

    return {
        "weather_available": context.is_available if context else False,
        "temperature_celsius": context.temperature_celsius if context else None,
        "condition": context.condition if context else None,
        "season_hint": season_hint,
        **guidance,
    }
