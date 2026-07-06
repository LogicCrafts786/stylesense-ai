"""Unit tests for src.tools.bundle_generator_tool."""

from __future__ import annotations

import pytest

from src.tools.bundle_generator_tool import generate_outfit_bundle
from src.utils.exceptions import AgentExecutionError, GeminiAPIError


class TestGenerateOutfitBundle:
    def test_raises_on_empty_candidates(self):
        with pytest.raises(AgentExecutionError):
            generate_outfit_bundle(candidate_products=[])

    def test_successful_generation(self, sample_products, mock_gemini_json_response, mocker):
        mock_gemini_json_response(
            {
                "selected_product_ids": ["TEST-001", "TEST-002"],
                "style_summary": "Sharp business casual look",
                "reasoning": "Navy and khaki pair well for the office.",
                "color_harmony_type": "neutral",
                "color_harmony_score": 0.85,
                "color_harmony_explanation": "Navy and khaki are classic neutrals.",
            }
        )
        mocker.patch(
            "src.tools.bundle_generator_tool.get_products_by_ids",
            return_value=[sample_products[0], sample_products[1]],
        )

        outfit = generate_outfit_bundle(
            candidate_products=sample_products, occasion="office", budget=150.0
        )

        assert outfit.item_count == 2
        assert outfit.style_summary == "Sharp business casual look"
        assert outfit.color_harmony.harmony_type == "neutral"

    def test_falls_back_to_first_candidates_if_llm_returns_no_valid_ids(
        self, sample_products, mock_gemini_json_response, mocker
    ):
        mock_gemini_json_response(
            {
                "selected_product_ids": ["NONEXISTENT-ID"],
                "style_summary": "Fallback look",
                "reasoning": "Fallback reasoning.",
            }
        )
        mocker.patch("src.tools.bundle_generator_tool.get_products_by_ids", return_value=[])

        outfit = generate_outfit_bundle(candidate_products=sample_products, occasion="office")
        assert outfit.item_count == min(3, len(sample_products))

    def test_gemini_failure_raises_agent_execution_error(self, sample_products, mocker):
        mocker.patch(
            "src.services.gemini_service.GeminiService.generate_structured_json",
            side_effect=GeminiAPIError("API down"),
        )
        with pytest.raises(AgentExecutionError):
            generate_outfit_bundle(candidate_products=sample_products, occasion="office")
