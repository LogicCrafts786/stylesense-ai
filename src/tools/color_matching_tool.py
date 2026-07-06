"""
Color matching tool.

Combines a rule-based color-theory baseline (fast, deterministic) with an
optional LLM-based refinement (nuanced, contextual) to score how well a
set of colors work together.
"""

from __future__ import annotations

from langchain_core.tools import tool

from src.models.outfit import ColorHarmonyAnalysis
from src.prompts.outfit_prompts import build_color_matching_prompt
from src.services.gemini_service import get_gemini_service
from src.utils.exceptions import GeminiAPIError
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Simplified color wheel neighbor map for rule-based fallback scoring.
# Not exhaustive — intended as a fast baseline before/instead of an LLM call.
_COLOR_WHEEL_NEIGHBORS: dict[str, list[str]] = {
    "red": ["orange", "pink", "maroon"],
    "orange": ["red", "yellow", "brown"],
    "yellow": ["orange", "green", "gold"],
    "green": ["yellow", "blue", "olive"],
    "blue": ["green", "purple", "navy"],
    "purple": ["blue", "pink", "lavender"],
    "pink": ["red", "purple", "white"],
    "brown": ["orange", "beige", "tan"],
    "black": ["white", "gray", "any"],
    "white": ["black", "gray", "any"],
    "gray": ["black", "white", "navy"],
    "navy": ["blue", "white", "gray"],
    "beige": ["brown", "white", "olive"],
}

_NEUTRALS = {"black", "white", "gray", "beige", "navy", "tan", "cream"}


def _rule_based_harmony_score(colors: list[str]) -> tuple[str, float]:
    """
    Compute a quick heuristic color harmony type and score without calling
    an LLM, used as a fast baseline or fallback if Gemini is unavailable.

    Args:
        colors: List of color names.

    Returns:
        Tuple of (harmony_type, score) where score is 0.0-1.0.
    """
    normalized = [c.lower().strip() for c in colors]

    if len(normalized) <= 1:
        return "monochromatic", 1.0

    unique_colors = set(normalized)
    if unique_colors.issubset(_NEUTRALS):
        return "neutral", 0.9

    non_neutral = [c for c in normalized if c not in _NEUTRALS]
    if len(non_neutral) <= 1:
        return "neutral", 0.85

    # Check if colors are wheel-adjacent (analogous) for the first pair found
    base = non_neutral[0]
    neighbors = set(_COLOR_WHEEL_NEIGHBORS.get(base, []))
    if any(c in neighbors for c in non_neutral[1:]):
        return "analogous", 0.75

    return "clashing", 0.4


def analyze_color_harmony(colors: list[str], *, use_llm_refinement: bool = True) -> ColorHarmonyAnalysis:
    """
    Analyze the color harmony of a palette, optionally refined by Gemini.

    Args:
        colors: List of color names to evaluate together.
        use_llm_refinement: If True, attempts an LLM call for a richer
            explanation; falls back to the rule-based result on failure.

    Returns:
        A ColorHarmonyAnalysis with harmony_type, score, and explanation.
    """
    if not colors:
        return ColorHarmonyAnalysis(
            harmony_type="unknown", score=0.5, explanation="No colors provided to analyze."
        )

    fallback_type, fallback_score = _rule_based_harmony_score(colors)

    if not use_llm_refinement:
        return ColorHarmonyAnalysis(
            harmony_type=fallback_type,
            score=fallback_score,
            explanation=f"Rule-based analysis: colors form a {fallback_type} palette.",
        )

    try:
        gemini = get_gemini_service()
        prompt = build_color_matching_prompt(colors)
        result = gemini.generate_structured_json(prompt)

        return ColorHarmonyAnalysis(
            harmony_type=result.get("harmony_type", fallback_type),
            score=float(result.get("score", fallback_score)),
            explanation=result.get("explanation", "No explanation provided."),
        )
    except GeminiAPIError as exc:
        logger.warning(f"LLM color refinement failed, using rule-based fallback: {exc}")
        return ColorHarmonyAnalysis(
            harmony_type=fallback_type,
            score=fallback_score,
            explanation=f"Rule-based analysis: colors form a {fallback_type} palette.",
        )


@tool
def color_matching_tool_fn(colors: list[str]) -> dict:
    """
    LangChain tool interface: analyze how well a list of colors work
    together and return a harmony type, score, and explanation.

    Args:
        colors: List of color names (e.g., ["navy", "white", "tan"]).

    Returns:
        Dictionary with harmony_type, score, and explanation.
    """
    analysis = analyze_color_harmony(colors)
    return {
        "harmony_type": analysis.harmony_type,
        "score": analysis.score,
        "explanation": analysis.explanation,
    }
