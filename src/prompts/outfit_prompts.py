"""
Prompt templates for outfit recommendation, bundle assembly, and
color/style reasoning.
"""

from __future__ import annotations

from src.models.product import Product


def build_outfit_reasoning_prompt(
    *,
    candidate_products: list[Product],
    occasion: str | None,
    budget: float | None,
    weather_description: str | None,
    preferred_colors: list[str] | None,
    preferred_styles: list[str] | None,
) -> str:
    """
    Build a prompt asking Gemini to select and justify a coherent outfit
    from a list of candidate products.

    Args:
        candidate_products: Pre-filtered products (already budget/category
            filtered) for the LLM to choose from.
        occasion: The stated or inferred occasion.
        budget: The user's stated budget, if any.
        weather_description: Weather/season context, if available.
        preferred_colors: User's known color preferences.
        preferred_styles: User's known style preferences.

    Returns:
        A formatted prompt string requesting structured JSON outfit output.
    """
    catalog_lines = "\n".join(
        f"- id={p.product_id} | {p.name} | category={p.category.value} | "
        f"price=${p.price:.2f} | colors={p.colors} | style_tags={p.style_tags}"
        for p in candidate_products
    )

    context_lines = [
        f"Occasion: {occasion or 'not specified'}",
        f"Budget: {f'${budget:.2f}' if budget else 'not specified'}",
        f"Weather/season: {weather_description or 'not specified'}",
        f"User's preferred colors: {preferred_colors or 'none known'}",
        f"User's preferred styles: {preferred_styles or 'none known'}",
    ]

    return f"""\
Given the following candidate products and context, assemble ONE complete, \
cohesive outfit (aim for top + bottom OR dress, plus shoes, plus an \
optional accessory). Only choose from the candidate list below — do not \
invent items.

CONTEXT:
{chr(10).join(context_lines)}

CANDIDATE PRODUCTS:
{catalog_lines}

Respond with ONLY valid JSON in this exact format, no markdown fences:
{{
  "selected_product_ids": ["<id>", "<id>", ...],
  "style_summary": "<one sentence describing the overall look>",
  "reasoning": "<2-4 sentences explaining color/style/occasion/budget fit>",
  "color_harmony_type": "<e.g. complementary, analogous, monochromatic, neutral>",
  "color_harmony_score": <float 0.0-1.0>,
  "color_harmony_explanation": "<1-2 sentences on why these colors work together>"
}}
"""


def build_alternative_outfit_prompt(
    *,
    original_outfit_summary: str,
    candidate_products: list[Product],
    reason_for_alternative: str,
) -> str:
    """
    Build a prompt requesting an alternative outfit variant (e.g., cheaper,
    different style, different color palette).

    Args:
        original_outfit_summary: Short description of the original outfit.
        candidate_products: New candidate pool to choose from.
        reason_for_alternative: Why an alternative is needed (e.g., "user
            wants a cheaper option", "user disliked the color palette").

    Returns:
        A formatted prompt string.
    """
    catalog_lines = "\n".join(
        f"- id={p.product_id} | {p.name} | category={p.category.value} | "
        f"price=${p.price:.2f} | colors={p.colors}"
        for p in candidate_products
    )

    return f"""\
The user was originally shown this outfit: {original_outfit_summary}

Reason an alternative is needed: {reason_for_alternative}

Choose a DIFFERENT cohesive outfit from the candidates below that addresses \
the reason above.

CANDIDATE PRODUCTS:
{catalog_lines}

Respond with ONLY valid JSON in this exact format, no markdown fences:
{{
  "selected_product_ids": ["<id>", "<id>", ...],
  "style_summary": "<one sentence>",
  "reasoning": "<why this addresses the stated reason>"
}}
"""


def build_color_matching_prompt(colors: list[str]) -> str:
    """
    Build a prompt asking Gemini to analyze the color harmony of a given
    palette using color theory principles.

    Args:
        colors: List of color names to analyze together.

    Returns:
        A formatted prompt string requesting structured JSON analysis.
    """
    return f"""\
Analyze the color harmony of this palette using standard color theory: {colors}

Respond with ONLY valid JSON in this exact format, no markdown fences:
{{
  "harmony_type": "<complementary | analogous | monochromatic | triadic | neutral | clashing>",
  "score": <float 0.0-1.0, where 1.0 is excellent harmony>,
  "explanation": "<1-2 sentences explaining the relationship between these colors>"
}}
"""
