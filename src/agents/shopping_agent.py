"""
Shopping agent orchestrator.

Provides the high-level, user-facing interface to the LangGraph shopping
agent, wiring together conversation memory, the compiled graph, and
response delivery. This is the module the Streamlit UI layer calls into.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.agents.agent_state import AgentState, create_initial_state
from src.agents.graph_builder import build_shopping_agent_graph
from src.memory.conversation_memory import ConversationMemory, get_conversation_memory_store
from src.models.conversation import MessageRole
from src.models.outfit import Outfit
from src.utils.exceptions import AgentExecutionError
from src.utils.logger import get_logger
from src.utils.validators import validate_user_message

logger = get_logger(__name__)


@dataclass
class AgentResponse:
    """
    Structured result of a single agent turn, returned to the UI layer.

    Attributes:
        text: The final natural-language response to display.
        outfit: An assembled Outfit object, if one was generated this turn.
        comparison_result: Structured comparison data, if generated this turn.
        review_summary: Structured review summary data, if generated this turn.
        image_analysis_result: Structured image analysis data, if an image was processed.
        detected_intent: The classified intent for this turn (for UI debugging/badges).
    """

    text: str
    outfit: Outfit | None = None
    comparison_result: dict | None = None
    review_summary: dict | None = None
    image_analysis_result: dict | None = None
    detected_intent: str | None = None


class ShoppingAgent:
    """
    High-level orchestrator for a single user's shopping conversation.

    Wraps the compiled LangGraph app with conversation memory management,
    input validation, and consistent error handling so the UI layer can
    call a single `handle_message` method per turn.
    """

    def __init__(self) -> None:
        self._graph = build_shopping_agent_graph()
        logger.info("ShoppingAgent initialized with compiled LangGraph.")

    def handle_message(
        self,
        *,
        user_id: str,
        user_message: str,
        image_bytes: bytes | None = None,
    ) -> AgentResponse:
        """
        Process a single user turn end-to-end: validate input, run the
        graph, update conversation memory, and return a structured response.

        Args:
            user_id: Unique session/user identifier for memory scoping.
            user_message: The raw user input text for this turn.
            image_bytes: Optional raw bytes of an uploaded image.

        Returns:
            An AgentResponse containing the final text and any structured
            results (outfit, comparison, review summary, image analysis).

        Raises:
            AgentExecutionError: If the graph fails to produce any usable output.
        """
        validated_message = validate_user_message(user_message)

        memory_store = get_conversation_memory_store()
        conversation_memory: ConversationMemory = memory_store.get_or_create(user_id)

        conversation_memory.add_user_message(validated_message, image_attached=bool(image_bytes))

        initial_state: AgentState = create_initial_state(
            user_id=user_id,
            user_message=validated_message,
            image_bytes=image_bytes,
            conversation_context=conversation_memory.get_context_as_text(),
        )

        try:
            final_state: AgentState = self._graph.invoke(initial_state)
        except Exception as exc:
            logger.error(f"Graph execution failed for user_id={user_id}: {exc}")
            raise AgentExecutionError("The shopping agent encountered an unexpected error.", details=str(exc)) from exc

        response_text = final_state.get("final_response") or (
            "I'm sorry, I wasn't able to generate a response. Could you rephrase your request?"
        )

        conversation_memory.add_assistant_message(
            response_text,
            detected_intent=final_state.get("detected_intent"),
            tool_calls_made=final_state.get("tool_calls_made", []),
        )

        logger.info(
            f"Turn complete for user_id={user_id}: intent={final_state.get('detected_intent')}, "
            f"tools_used={final_state.get('tool_calls_made', [])}"
        )

        return AgentResponse(
            text=response_text,
            outfit=final_state.get("recommended_outfit"),
            comparison_result=final_state.get("comparison_result"),
            review_summary=final_state.get("review_summary"),
            image_analysis_result=final_state.get("image_analysis_result"),
            detected_intent=final_state.get("detected_intent"),
        )

    def reset_conversation(self, user_id: str) -> None:
        """
        Clear a user's conversation history, starting a fresh session while
        preserving their long-term profile preferences.

        Args:
            user_id: The session/user identifier to reset.
        """
        memory_store = get_conversation_memory_store()
        conversation_memory = memory_store.get_or_create(user_id)
        conversation_memory.clear()
        logger.info(f"Conversation reset for user_id={user_id}.")


_agent_instance: ShoppingAgent | None = None


def get_shopping_agent() -> ShoppingAgent:
    """Return a lazily-initialized singleton ShoppingAgent instance."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = ShoppingAgent()
    return _agent_instance
