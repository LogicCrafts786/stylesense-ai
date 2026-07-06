"""
Conversation and message schemas for StyleSense AI.

These models represent the multi-turn chat history maintained by
conversation_memory.py and rendered by the Streamlit chat interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from uuid import uuid4


class MessageRole(str, Enum):
    """Role of the message sender."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class Message:
    """
    Represents a single message in the conversation.

    Attributes:
        message_id: Unique identifier for the message.
        role: Who sent the message (user, assistant, system).
        content: The text content of the message.
        image_attached: Whether an image was attached to this message.
        timestamp: When the message was created.
        metadata: Arbitrary extra data (e.g., detected_intent, tool_calls).
    """

    role: MessageRole
    content: str
    message_id: str = field(default_factory=lambda: str(uuid4()))
    image_attached: bool = False
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize the message to a plain dictionary."""
        return {
            "message_id": self.message_id,
            "role": self.role.value,
            "content": self.content,
            "image_attached": self.image_attached,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Message":
        """Deserialize a Message from a plain dictionary."""
        return cls(
            message_id=data.get("message_id", str(uuid4())),
            role=MessageRole(data["role"]),
            content=data["content"],
            image_attached=data.get("image_attached", False),
            timestamp=datetime.fromisoformat(data["timestamp"])
            if data.get("timestamp")
            else datetime.utcnow(),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Conversation:
    """
    Represents a full conversation session (a sequence of Messages).

    Attributes:
        conversation_id: Unique identifier for this conversation/session.
        messages: Ordered list of Message objects.
        created_at: When the conversation started.
    """

    conversation_id: str = field(default_factory=lambda: str(uuid4()))
    messages: list[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def add_message(self, role: MessageRole, content: str, **kwargs) -> Message:
        """
        Append a new message to the conversation.

        Args:
            role: The sender's role.
            content: The message text.
            **kwargs: Additional Message fields (image_attached, metadata).

        Returns:
            The newly created Message instance.
        """
        message = Message(role=role, content=content, **kwargs)
        self.messages.append(message)
        return message

    def get_recent_messages(self, limit: int) -> list[Message]:
        """
        Retrieve the most recent `limit` messages, preserving chronological order.

        Args:
            limit: Maximum number of recent messages to return.

        Returns:
            A list of the most recent Message objects, oldest first.
        """
        if limit <= 0:
            return []
        return self.messages[-limit:]

    def to_langchain_format(self, limit: int | None = None) -> list[dict]:
        """
        Convert conversation history into a simple role/content dict list
        compatible with LangChain message constructors.

        Args:
            limit: Optional cap on how many recent messages to include.

        Returns:
            List of {"role": ..., "content": ...} dictionaries.
        """
        messages = self.get_recent_messages(limit) if limit else self.messages
        return [{"role": msg.role.value, "content": msg.content} for msg in messages]
