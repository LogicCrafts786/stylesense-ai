"""Unit tests for src.agents.shopping_agent.ShoppingAgent."""

from __future__ import annotations

import pytest

from src.utils.exceptions import ValidationError


@pytest.fixture
def mock_compiled_graph(mocker):
    """Patch build_shopping_agent_graph so ShoppingAgent doesn't build a real LangGraph."""
    mock_graph = mocker.MagicMock()
    mocker.patch(
        "src.agents.shopping_agent.build_shopping_agent_graph", return_value=mock_graph
    )
    return mock_graph


class TestShoppingAgent:
    def test_handle_message_returns_agent_response(self, mock_compiled_graph):
        from src.agents.shopping_agent import ShoppingAgent

        mock_compiled_graph.invoke.return_value = {
            "final_response": "Here's a great outfit for you!",
            "detected_intent": "recommend_outfit",
            "recommended_outfit": None,
            "comparison_result": None,
            "review_summary": None,
            "image_analysis_result": None,
            "tool_calls_made": ["intent_router", "entity_extraction"],
        }

        agent = ShoppingAgent()
        response = agent.handle_message(
            user_id="test-session", user_message="I need an outfit for a wedding"
        )

        assert response.text == "Here's a great outfit for you!"
        assert response.detected_intent == "recommend_outfit"

    def test_handle_message_validates_input(self, mock_compiled_graph):
        from src.agents.shopping_agent import ShoppingAgent

        agent = ShoppingAgent()
        with pytest.raises(ValidationError):
            agent.handle_message(user_id="test-session", user_message="   ")

    def test_handle_message_falls_back_on_missing_final_response(self, mock_compiled_graph):
        from src.agents.shopping_agent import ShoppingAgent

        mock_compiled_graph.invoke.return_value = {
            "final_response": None,
            "detected_intent": "general_chat",
            "tool_calls_made": [],
        }

        agent = ShoppingAgent()
        response = agent.handle_message(user_id="test-session", user_message="Hello")
        assert "wasn't able to generate" in response.text

    def test_reset_conversation_clears_memory(self, mock_compiled_graph):
        from src.agents.shopping_agent import ShoppingAgent
        from src.memory.conversation_memory import get_conversation_memory_store

        agent = ShoppingAgent()
        mock_compiled_graph.invoke.return_value = {
            "final_response": "Hi!",
            "detected_intent": "general_chat",
            "tool_calls_made": [],
        }
        agent.handle_message(user_id="reset-test-session", user_message="Hello")

        store = get_conversation_memory_store()
        memory_before = store.get_or_create("reset-test-session")
        assert memory_before.message_count() > 0

        agent.reset_conversation("reset-test-session")
        memory_after = store.get_or_create("reset-test-session")
        assert memory_after.message_count() == 0
