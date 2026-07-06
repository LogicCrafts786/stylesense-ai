"""Unit tests for src.tools.budget_filter_tool."""

from __future__ import annotations

import pytest

from src.tools.budget_filter_tool import (
    check_outfit_affordability,
    filter_products_by_budget,
    suggest_category_allocation,
)
from src.utils.exceptions import BudgetConstraintError, ValidationError


class TestFilterProductsByBudget:
    def test_filters_correctly(self, sample_products):
        results = filter_products_by_budget(sample_products, max_budget=60.0)
        assert all(p.price <= 60.0 for p in results)
        assert len(results) == 2  # excludes the $200 sneakers

    def test_no_products_within_budget_raises(self, sample_products):
        with pytest.raises(BudgetConstraintError):
            filter_products_by_budget(sample_products, max_budget=1.0)

    def test_invalid_budget_raises_validation_error(self, sample_products):
        with pytest.raises(ValidationError):
            filter_products_by_budget(sample_products, max_budget=-5.0)


class TestSuggestCategoryAllocation:
    def test_allocations_sum_to_budget(self):
        allocation = suggest_category_allocation(200.0, ["top", "bottom", "shoes"])
        assert abs(sum(allocation.values()) - 200.0) < 0.01

    def test_unknown_category_gets_default_weight(self):
        allocation = suggest_category_allocation(100.0, ["unknown_category"])
        assert "unknown_category" in allocation
        assert allocation["unknown_category"] > 0


class TestCheckOutfitAffordability:
    def test_within_budget(self):
        result = check_outfit_affordability([40.0, 30.0], budget=100.0)
        assert result["is_within_budget"] is True
        assert result["difference"] == 30.0

    def test_over_budget(self):
        result = check_outfit_affordability([80.0, 50.0], budget=100.0)
        assert result["is_within_budget"] is False
        assert result["difference"] == -30.0
