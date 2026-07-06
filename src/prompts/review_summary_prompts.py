"""
Prompt templates for review summarization and product comparison tasks.
"""

from __future__ import annotations

from src.models.product import Product


def build_review_summary_prompt(product_name: str, review_texts: list[str]) -> str:
    """
    Build a prompt requesting a structured pros/cons summary of product reviews.

    Args:
        product_name: The name of the product being reviewed.
        review_texts: Raw review text snippets scraped or loaded from sample data.

    Returns:
        A formatted prompt string requesting structured JSON output.
    """
    numbered_reviews = "\n".join(f"{i + 1}. {text}" for i, text in enumerate(review_texts))

    return f"""\
Summarize the following customer reviews for the product "{product_name}". \
Identify recurring themes — do not simply restate individual reviews.

REVIEWS:
{numbered_reviews}

Respond with ONLY valid JSON in this exact format, no markdown fences:
{{
  "overall_sentiment": "<positive | mixed | negative>",
  "pros": ["<theme>", ...],
  "cons": ["<theme>", ...],
  "fit_notes": "<any recurring comments about sizing/fit, or null>",
  "quality_notes": "<any recurring comments about material/durability, or null>",
  "summary": "<2-3 sentence overall summary>"
}}
"""


def build_product_comparison_prompt(products: list[Product], user_priority: str | None = None) -> str:
    """
    Build a prompt requesting a structured side-by-side comparison of products.

    Args:
        products: The list of Product objects to compare (2 or more).
        user_priority: What the user cares most about (e.g., "value for money",
            "quality", "versatility"). If None, a balanced comparison is requested.

    Returns:
        A formatted prompt string requesting structured JSON output.
    """
    product_lines = "\n".join(
        f"- id={p.product_id} | {p.name} | brand={p.brand} | price=${p.price:.2f} | "
        f"rating={p.rating or 'N/A'} ({p.review_count} reviews) | style_tags={p.style_tags}"
        for p in products
    )

    priority_line = (
        f"The user's stated priority is: {user_priority}."
        if user_priority
        else "The user has not stated a specific priority — provide a balanced comparison."
    )

    return f"""\
Compare the following products for a shopper trying to decide between them. \
{priority_line}

PRODUCTS:
{product_lines}

Respond with ONLY valid JSON in this exact format, no markdown fences:
{{
  "comparison_table": [
    {{
      "product_id": "<id>",
      "value_for_money": "<brief assessment>",
      "style_fit": "<brief assessment>",
      "notable_strength": "<one key strength>",
      "notable_weakness": "<one key weakness or null>"
    }}
  ],
  "recommendation": "<product_id of the recommended choice>",
  "recommendation_reasoning": "<2-3 sentences explaining the recommendation>"
}}
"""  
