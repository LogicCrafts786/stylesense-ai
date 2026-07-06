"""
LangGraph state schema for the StyleSense shopping agent.

This TypedDict is threaded through every node in the graph. Each node
reads relevant fields and returns a partial dict of updates, which
LangGraph merges into the running state.
"""

from __future__ import annotations

from typing import Any, TypedDict

from src.models.outfit import Outfit
from src.models.product import Product


class AgentState(TypedDict, total=False):
    """
    Shared state object passed between all LangGraph nodes.

    Attributes:
        user_id: Session/user identifier, used for memory lookups.
        user_message: The raw text message from the user this turn.
        image_bytes: Raw bytes of an uploaded image, if any.
        conversation_context: Formatted transcript of recent chat history.
        detected_intent: Classified intent (recommend_outfit, compare_products, etc.).
        intent_confidence: Confidence score for the detected intent.
        extracted_entities: Dict of budget, occasion, colors, styles, weather, etc.
        image_analysis_result: Structured output from Gemini Vision, if an image was analyzed.
        weather_context: Dict of weather/season guidance for the request.
        candidate_products: List of Product objects retrieved as candidates.
        recommended_outfit: The assembled Outfit object, if generated.
        comparison_result: Structured comparison output, if a comparison was requested.
        review_summary: Structured review summary output, if requested.
        final_response: The final natural-language response to show the user.
        error: Any error message to surface gracefully, if a node failed.
        tool_calls_made: List of tool names invoked during this turn (for observability/logging).
    """

    user_id: str
    user_message: str
    image_bytes: bytes | None
    conversation_context: str
    detected_intent: str | None
    intent_confidence: float | None
    extracted_entities: dict[str, Any]
    image_analysis_result: dict[str, Any] | None
    weather_context: dict[str, Any] | None
    candidate_products: list[Product]
    recommended_outfit: Outfit | None
    comparison_result: dict[str, Any] | None
    review_summary: dict[str, Any] | None
    final_response: str | None
    error: str | None
    tool_calls_made: list[str]


def create_initial_state(
    *,
    user_id: str,
    user_message: str,
    image_bytes: bytes | None = None,
    conversation_context: str = "",
) -> AgentState:
    """
    Build a fresh AgentState for the start of a new graph invocation.

    Args:
        user_id: The session/user identifier.
        user_message: The user's raw input text for this turn.
        image_bytes: Optional raw bytes of an uploaded image.
        conversation_context: Formatted recent chat transcript.

    Returns:
        A fully initialized AgentState dictionary with safe defaults.
    """
    return AgentState(
        user_id=user_id,
        user_message=user_message,
        image_bytes=image_bytes,
        conversation_context=conversation_context,
        detected_intent=None,
        intent_confidence=None,
        extracted_entities={},
        image_analysis_result=None,
        weather_context=None,
        candidate_products=[],
        recommended_outfit=None,
        comparison_result=None,
        review_summary=None,
        final_response=None,
        error=None,
        tool_calls_made=[],
    )
