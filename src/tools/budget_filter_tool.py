"""
Budget filtering tool.

Applies hard budget constraints to product/outfit candidates and offers
budget-aware suggestions (e.g., "how many items you can afford", or
"suggested per-item allocation to fit within total budget").
"""

from __future__ import annotations

from langchain_core.tools import tool

from src.models.product import Product
from src.utils.exceptions import BudgetConstraintError
from src.utils.logger import get_logger
from src.utils.validators import validate_budget

logger = get_logger(__name__)

# Reasonable default allocation percentages per category for an outfit budget,
# used to suggest per-item spending caps when assembling a bundle.
_DEFAULT_ALLOCATION = {
    "top": 0.25,
    "bottom": 0.25,
    "dress": 0.45,
    "outerwear": 0.30,
    "shoes": 0.30,
    "accessory": 0.10,
    "bag": 0.15,
    "jewelry": 0.08,
}


def filter_products_by_budget(products: list[Product], max_budget: float) -> list[Product]:
    """
    Filter a list of products to only those within the given budget.

    Args:
        products: Candidate Product list.
        max_budget: Maximum acceptable price per item.

    Returns:
        Filtered list of products within budget.

    Raises:
        BudgetConstraintError: If no products fall within the budget at all.
    """
    validated_budget = validate_budget(max_budget)
    within_budget = [p for p in products if p.matches_budget(validated_budget)]

    if not within_budget:
        raise BudgetConstraintError(
            f"No products found within the budget of ${validated_budget:.2f}.",
            details=f"Checked {len(products)} candidate products.",
        )

    logger.debug(f"{len(within_budget)}/{len(products)} products within budget ${validated_budget:.2f}")
    return within_budget


def suggest_category_allocation(total_budget: float, categories: list[str]) -> dict[str, float]:
    """
    Suggest a per-category spending cap that sums to the total budget, based
    on typical allocation ratios (e.g., shoes/outerwear get more, accessories less).

    Args:
        total_budget: The user's total budget for the outfit/bundle.
        categories: List of category strings to allocate budget across.

    Returns:
        Dictionary mapping category -> suggested max spend for that category.
    """
    validated_budget = validate_budget(total_budget)

    relevant_weights = {cat: _DEFAULT_ALLOCATION.get(cat, 0.20) for cat in categories}
    total_weight = sum(relevant_weights.values()) or 1.0

    return {
        cat: round((weight / total_weight) * validated_budget, 2)
        for cat, weight in relevant_weights.items()
    }


def check_outfit_affordability(item_prices: list[float], budget: float) -> dict:
    """
    Check whether a proposed set of item prices fits within budget, and by
    how much it's over/under.

    Args:
        item_prices: List of prices for the proposed outfit items.
        budget: The user's stated total budget.

    Returns:
        Dictionary with keys: total_price, budget, is_within_budget,
        difference (positive = under budget, negative = over budget).
    """
    validated_budget = validate_budget(budget)
    total_price = round(sum(item_prices), 2)
    difference = round(validated_budget - total_price, 2)

    return {
        "total_price": total_price,
        "budget": validated_budget,
        "is_within_budget": total_price <= validated_budget,
        "difference": difference,
    }


@tool
def budget_filter_tool_fn(product_ids_with_prices: dict[str, float], max_budget: float) -> dict:
    """
    LangChain tool interface: check affordability of a set of products
    against a budget.

    Args:
        product_ids_with_prices: Mapping of product_id -> price.
        max_budget: The user's stated budget.

    Returns:
        Affordability check result dictionary.
    """
    return check_outfit_affordability(list(product_ids_with_prices.values()), max_budget)
