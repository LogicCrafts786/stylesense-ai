"""Unit tests for src.tools.color_matching_tool."""

from __future__ import annotations

from src.tools.color_matching_tool import _rule_based_harmony_score, analyze_color_harmony
from src.utils.exceptions import GeminiAPIError


class TestRuleBasedHarmonyScore:
    def test_single_color_is_monochromatic(self):
        harmony_type, score = _rule_based_harmony_score(["navy"])
        assert harmony_type == "monochromatic"
        assert score == 1.0

    def test_all_neutrals_scores_high(self):
        harmony_type, score = _rule_based_harmony_score(["black", "white", "gray"])
        assert harmony_type == "neutral"
        assert score >= 0.8

    def test_wheel_adjacent_colors_are_analogous(self):
        harmony_type, _ = _rule_based_harmony_score(["red", "orange"])
        assert harmony_type == "analogous"

    def test_unrelated_colors_flagged_as_clashing(self):
        harmony_type, score = _rule_based_harmony_score(["red", "purple"])
        assert harmony_type == "clashing"
        assert score < 0.5


class TestAnalyzeColorHarmony:
    def test_empty_colors_returns_unknown(self):
        result = analyze_color_harmony([])
        assert result.harmony_type == "unknown"

    def test_llm_disabled_uses_rule_based_fallback(self):
        result = analyze_color_harmony(["navy", "white"], use_llm_refinement=False)
        assert result.harmony_type in {"neutral", "analogous", "monochromatic", "clashing"}

    def test_llm_success_path(self, mock_gemini_json_response):
        mock_gemini_json_response(
            {
                "harmony_type": "complementary",
                "score": 0.95,
                "explanation": "These colors sit opposite on the color wheel.",
            }
        )
        result = analyze_color_harmony(["blue", "orange"], use_llm_refinement=True)
        assert result.harmony_type == "complementary"
        assert result.score == 0.95

    def test_llm_failure_falls_back_to_rule_based(self, mocker):
        mocker.patch(
            "src.services.gemini_service.GeminiService.generate_structured_json",
            side_effect=GeminiAPIError("API down"),
        )
        result = analyze_color_harmony(["navy", "white"], use_llm_refinement=True)
        # Should not raise; falls back gracefully
        assert result.harmony_type is not None
