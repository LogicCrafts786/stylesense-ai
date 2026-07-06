"""
Bundle generator tool.

The capstone tool that ties everything together: given filtered candidate
products and context (occasion, budget, weather, preferences), it calls
Gemini to assemble a cohesive Outfit, attaches color harmony analysis, and
can generate alternative variants.
"""

from __future__ import annotations

from uuid import uuid4

from langchain_core.tools import tool

from src.models.outfit import ColorHarmonyAnalysis, Outfit
from src.models.product import Product
from src.prompts.outfit_prompts import build_alternative_outfit_prompt, build_outfit_reasoning_prompt
from src.services.gemini_service import get_gemini_service
from src.tools.color_matching_tool import analyze_color_harmony
from src.tools.product_search_tool import get_products_by_ids
from src.utils.exceptions import AgentExecutionError, GeminiAPIError
from src.utils.logger import get_logger

logger = get_logger(__name__)


def generate_outfit_bundle(
    *,
    candidate_products: list[Product],
    occasion: str | None = None,
    budget: float | None = None,
    weather_description: str | None = None,
    preferred_colors: list[str] | None = None,
    preferred_styles: list[str] | None = None,
) -> Outfit:
    """
    Assemble a complete, cohesive Outfit from candidate products using
    Gemini reasoning, then attach a color harmony analysis.

    Args:
        candidate_products: Pre-filtered products (category/budget already applied).
        occasion: The occasion context.
        budget: The user's stated budget.
        weather_description: Weather/season context string.
        preferred_colors: Known user color preferences.
        preferred_styles: Known user style preferences.

    Returns:
        A fully assembled Outfit object with reasoning and color harmony.

    Raises:
        AgentExecutionError: If no candidates are provided or Gemini fails
            to return a usable selection.
    """
    if not candidate_products:
        raise AgentExecutionError("Cannot generate an outfit: no candidate products were provided.")

    prompt = build_outfit_reasoning_prompt(
        candidate_products=candidate_products,
        occasion=occasion,
        budget=budget,
        weather_description=weather_description,
        preferred_colors=preferred_colors,
        preferred_styles=preferred_styles,
    )

    try:
        gemini = get_gemini_service()
        result = gemini.generate_structured_json(prompt, temperature=0.5)
    except GeminiAPIError as exc:
        logger.error(f"Outfit generation LLM call failed: {exc}")
        raise AgentExecutionError("Failed to generate outfit recommendation.", details=str(exc)) from exc

    selected_ids = result.get("selected_product_ids", [])
    selected_products = get_products_by_ids(selected_ids)

    if not selected_products:
        logger.warning("Gemini returned no valid product IDs; falling back to first 3 candidates.")
        selected_products = candidate_products[:3]

    all_colors = [color for p in selected_products for color in p.colors]
    color_harmony = ColorHarmonyAnalysis(
        harmony_type=result.get("color_harmony_type", "unknown"),
        score=float(result.get("color_harmony_score", 0.5)),
        explanation=result.get("color_harmony_explanation", ""),
    )

    # Cross-check with rule-based analysis if the LLM omitted color harmony fields
    if not result.get("color_harmony_type"):
        color_harmony = analyze_color_harmony(all_colors, use_llm_refinement=False)

    outfit = Outfit(
        outfit_id=str(uuid4()),
        items=selected_products,
        occasion=occasion,
        weather_context=weather_description,
        style_summary=result.get("style_summary", ""),
        reasoning=result.get("reasoning", ""),
        color_harmony=color_harmony,
    )

    logger.info(
        f"Generated outfit {outfit.outfit_id} with {outfit.item_count} items, "
        f"total price=${outfit.total_price:.2f}"
    )
    return outfit


def generate_alternative_outfit(
    original_outfit: Outfit,
    candidate_products: list[Product],
    reason_for_alternative: str,
) -> Outfit:
    """
    Generate an alternative outfit variant addressing a specific reason
    (e.g., "over budget", "user disliked the colors").

    Args:
        original_outfit: The previously generated Outfit to diverge from.
        candidate_products: New candidate pool (should already reflect any
            new constraints, e.g., a lower price cap).
        reason_for_alternative: Human-readable reason driving the change.

    Returns:
        A new Outfit object representing the alternative.

    Raises:
        AgentExecutionError: If Gemini fails to produce a usable alternative.
    """
    prompt = build_alternative_outfit_prompt(
        original_outfit_summary=str(original_outfit),
        candidate_products=candidate_products,
        reason_for_alternative=reason_for_alternative,
    )

    try:
        gemini = get_gemini_service()
        result = gemini.generate_structured_json(prompt, temperature=0.5)
    except GeminiAPIError as exc:
        raise AgentExecutionError("Failed to generate alternative outfit.", details=str(exc)) from exc

    selected_products = get_products_by_ids(result.get("selected_product_ids", []))
    if not selected_products:
        selected_products = candidate_products[:3]

    return Outfit(
        outfit_id=str(uuid4()),
        items=selected_products,
        occasion=original_outfit.occasion,
        weather_context=original_outfit.weather_context,
        style_summary=result.get("style_summary", ""),
        reasoning=result.get("reasoning", ""),
    )


@tool
def bundle_generator_tool_fn(
    candidate_product_ids: list[str],
    occasion: str | None = None,
    budget: float | None = None,
) -> dict:
    """
    LangChain tool interface: generate a cohesive outfit bundle from a set
    of candidate product IDs.

    Args:
        candidate_product_ids: Product IDs to choose from.
        occasion: Optional occasion context.
        budget: Optional budget constraint.

    Returns:
        Serialized Outfit dictionary, or an error dict on failure.
    """
    candidates = get_products_by_ids(candidate_product_ids)
    try:
        outfit = generate_outfit_bundle(candidate_products=candidates, occasion=occasion, budget=budget)
        return outfit.to_dict()
    except AgentExecutionError as exc:
        return {"error": str(exc)}
