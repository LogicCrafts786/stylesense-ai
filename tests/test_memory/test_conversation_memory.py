"""Unit tests for src.memory.conversation_memory."""

from __future__ import annotations

import pytest

from src.memory.conversation_memory import ConversationMemory, ConversationMemoryStore
from src.models.conversation import MessageRole
from src.utils.exceptions import MemoryError_


class TestConversationMemory:
    def test_add_user_message(self):
        memory = ConversationMemory()
        message = memory.add_user_message("Hello there")
        assert message.role == MessageRole.USER
        assert message.content == "Hello there"
        assert memory.message_count() == 1

    def test_add_assistant_message(self):
        memory = ConversationMemory()
        memory.add_assistant_message("Hi! How can I help?")
        assert memory.message_count() == 1
        assert memory.conversation.messages[0].role == MessageRole.ASSISTANT

    def test_get_context_window_respects_limit(self, monkeypatch):
        memory = ConversationMemory()
        for i in range(30):
            memory.add_user_message(f"Message {i}")

        window = memory.get_context_window()
        assert len(window) <= memory._settings.max_conversation_history

    def test_get_context_as_text_empty_conversation(self):
        memory = ConversationMemory()
        assert memory.get_context_as_text() == "(no prior conversation)"

    def test_get_context_as_text_formats_correctly(self):
        memory = ConversationMemory()
        memory.add_user_message("Hi")
        memory.add_assistant_message("Hello!")
        text = memory.get_context_as_text()
        assert "User: Hi" in text
        assert "Assistant: Hello!" in text

    def test_clear_resets_conversation(self):
        memory = ConversationMemory()
        memory.add_user_message("Hi")
        original_id = memory.conversation.conversation_id
        memory.clear()
        assert memory.message_count() == 0
        assert memory.conversation.conversation_id != original_id

    def test_get_last_user_message(self):
        memory = ConversationMemory()
        memory.add_user_message("First")
        memory.add_assistant_message("Reply")
        memory.add_user_message("Second")
        last = memory.get_last_user_message()
        assert last.content == "Second"

    def test_get_last_user_message_none_when_empty(self):
        memory = ConversationMemory()
        assert memory.get_last_user_message() is None


class TestConversationMemoryStore:
    def test_get_or_create_returns_same_instance(self):
        store = ConversationMemoryStore()
        mem1 = store.get_or_create("session-1")
        mem2 = store.get_or_create("session-1")
        assert mem1 is mem2

    def test_different_sessions_are_isolated(self):
        store = ConversationMemoryStore()
        mem1 = store.get_or_create("session-1")
        mem2 = store.get_or_create("session-2")
        mem1.add_user_message("Only in session 1")
        assert mem2.message_count() == 0

    def test_remove_session(self):
        store = ConversationMemoryStore()
        store.get_or_create("session-1")
        store.remove_session("session-1")
        assert store.active_session_count() == 0

    def test_remove_nonexistent_session_raises(self):
        store = ConversationMemoryStore()
        with pytest.raises(MemoryError_):
            store.remove_session("nonexistent")

    def test_active_session_count(self):
        store = ConversationMemoryStore()
        store.get_or_create("session-1")
        store.get_or_create("session-2")
        assert store.active_session_count() == 2
