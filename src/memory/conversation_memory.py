"""
Conversation memory management.

Maintains short-term, in-session multi-turn chat history for the shopping
agent. This is intentionally separate from long-term user preference
memory (see user_profile_memory.py) — conversation memory is scoped to a
single session and feeds directly into LLM context windows.
"""

from __future__ import annotations

from src.models.conversation import Conversation, Message, MessageRole
from src.utils.config import get_settings
from src.utils.exceptions import MemoryError_
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ConversationMemory:
    """
    Manages a single conversation's message history with a configurable
    sliding window to bound the context sent to the LLM.
    """

    def __init__(self, conversation: Conversation | None = None) -> None:
        self._settings = get_settings()
        self.conversation: Conversation = conversation or Conversation()

    def add_user_message(self, content: str, *, image_attached: bool = False, **metadata) -> Message:
        """
        Record a new user message in the conversation.

        Args:
            content: The user's message text.
            image_attached: Whether an image was attached to this turn.
            **metadata: Additional metadata to store (e.g., detected_intent).

        Returns:
            The created Message object.
        """
        return self.conversation.add_message(
            MessageRole.USER, content, image_attached=image_attached, metadata=metadata
        )

    def add_assistant_message(self, content: str, **metadata) -> Message:
        """
        Record a new assistant (agent) response in the conversation.

        Args:
            content: The assistant's response text.
            **metadata: Additional metadata to store (e.g., tool_calls_made).

        Returns:
            The created Message object.
        """
        return self.conversation.add_message(MessageRole.ASSISTANT, content, metadata=metadata)

    def get_context_window(self) -> list[Message]:
        """
        Retrieve the recent message window used for LLM context, bounded by
        MAX_CONVERSATION_HISTORY from settings.

        Returns:
            List of recent Message objects, oldest first.
        """
        return self.conversation.get_recent_messages(self._settings.max_conversation_history)

    def get_context_as_text(self) -> str:
        """
        Format the recent context window as a simple transcript string,
        suitable for embedding directly into a Gemini prompt.

        Returns:
            A newline-separated "Role: content" transcript string.
        """
        messages = self.get_context_window()
        if not messages:
            return "(no prior conversation)"

        lines = [f"{msg.role.value.capitalize()}: {msg.content}" for msg in messages]
        return "\n".join(lines)

    def clear(self) -> None:
        """Reset the conversation history, starting a fresh session."""
        logger.info(f"Clearing conversation memory for conversation_id={self.conversation.conversation_id}")
        self.conversation = Conversation()

    def get_last_user_message(self) -> Message | None:
        """
        Retrieve the most recent user message, if any.

        Returns:
            The last Message with role=USER, or None if no user messages exist.
        """
        user_messages = [m for m in self.conversation.messages if m.role == MessageRole.USER]
        return user_messages[-1] if user_messages else None

    def message_count(self) -> int:
        """Return the total number of messages in the conversation."""
        return len(self.conversation.messages)


class ConversationMemoryStore:
    """
    In-process registry of ConversationMemory instances keyed by session ID,
    allowing the Streamlit app to maintain separate histories per user
    session without a database dependency.

    Note: This is an in-memory store suitable for single-process deployments
    (e.g., a single Streamlit instance). For multi-instance production
    deployments, this should be backed by Redis or a similar shared store.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, ConversationMemory] = {}

    def get_or_create(self, session_id: str) -> ConversationMemory:
        """
        Retrieve an existing ConversationMemory for a session, or create a
        new one if none exists yet.

        Args:
            session_id: Unique identifier for the user's session
                (e.g., Streamlit session state ID).

        Returns:
            The ConversationMemory instance for this session.
        """
        if session_id not in self._sessions:
            logger.info(f"Creating new conversation memory for session_id={session_id}")
            self._sessions[session_id] = ConversationMemory()
        return self._sessions[session_id]

    def remove_session(self, session_id: str) -> None:
        """
        Remove a session's conversation memory entirely (e.g., on logout
        or explicit "start over").

        Args:
            session_id: The session identifier to remove.

        Raises:
            MemoryError_: If the session does not exist.
        """
        if session_id not in self._sessions:
            raise MemoryError_(f"No conversation memory found for session_id='{session_id}'.")
        del self._sessions[session_id]
        logger.info(f"Removed conversation memory for session_id={session_id}")

    def active_session_count(self) -> int:
        """Return the number of currently tracked sessions."""
        return len(self._sessions)


_store_instance: ConversationMemoryStore | None = None


def get_conversation_memory_store() -> ConversationMemoryStore:
    """Return a lazily-initialized singleton ConversationMemoryStore instance."""
    global _store_instance
    if _store_instance is None:
        _store_instance = ConversationMemoryStore()
    return _store_instance
