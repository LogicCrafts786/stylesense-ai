"""
Streamlit chat interface component.

Renders the main chat message history and input box, and dispatches
structured results (outfits, comparisons, review summaries, image
analysis) to their respective display components alongside each turn.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import streamlit as st

from src.agents.shopping_agent import AgentResponse, get_shopping_agent
from src.ui.image_uploader import render_image_uploader
from src.ui.outfit_display import (
    render_comparison_result,
    render_image_analysis_result,
    render_outfit_card,
    render_review_summary,
)
from src.utils.exceptions import AgentExecutionError, StyleSenseBaseError, ValidationError
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ChatTurn:
    """
    Represents one rendered turn in the chat UI: a user message and the
    corresponding assistant response (with any structured attachments).
    """

    user_text: str
    assistant_text: str
    outfit: object | None = None
    comparison_result: dict | None = None
    review_summary: dict | None = None
    image_analysis_result: dict | None = None


def initialize_chat_session_state() -> None:
    """Ensure chat_messages exists in Streamlit session state before rendering."""
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages: list[ChatTurn] = []


def render_chat_history() -> None:
    """Render all past chat turns, including any structured result cards."""
    for turn in st.session_state.chat_messages:
        with st.chat_message("user"):
            st.markdown(turn.user_text)

        with st.chat_message("assistant"):
            st.markdown(turn.assistant_text)
            if turn.image_analysis_result:
                render_image_analysis_result(turn.image_analysis_result)
            if turn.outfit:
                render_outfit_card(turn.outfit)
            if turn.comparison_result:
                render_comparison_result(turn.comparison_result)
            if turn.review_summary:
                render_review_summary(turn.review_summary)


def render_chat_interface(user_id: str) -> None:
    """
    Render the full chat interface: history, image uploader, and input box,
    handling new message submission end-to-end.

    Args:
        user_id: The current session's user identifier, used for memory scoping.
    """
    initialize_chat_session_state()

    st.markdown("### 💬 Chat with StyleSense")
    render_chat_history()

    uploaded_image_bytes = render_image_uploader()

    user_input = st.chat_input("Ask me for outfit ideas, comparisons, or style advice...")

    if user_input:
        _handle_new_message(user_id, user_input, uploaded_image_bytes)


def _handle_new_message(user_id: str, user_input: str, image_bytes: bytes | None) -> None:
    """
    Process a newly submitted user message: call the agent, render the
    response immediately, and append it to session history.

    Args:
        user_id: The current session's user identifier.
        user_input: The text the user typed.
        image_bytes: Optional uploaded image bytes for this turn.
    """
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking through your style..."):
            try:
                agent = get_shopping_agent()
                response: AgentResponse = agent.handle_message(
                    user_id=user_id, user_message=user_input, image_bytes=image_bytes
                )

                st.markdown(response.text)

                if response.image_analysis_result:
                    render_image_analysis_result(response.image_analysis_result)
                if response.outfit:
                    render_outfit_card(response.outfit)
                if response.comparison_result:
                    render_comparison_result(response.comparison_result)
                if response.review_summary:
                    render_review_summary(response.review_summary)

                st.session_state.chat_messages.append(
                    ChatTurn(
                        user_text=user_input,
                        assistant_text=response.text,
                        outfit=response.outfit,
                        comparison_result=response.comparison_result,
                        review_summary=response.review_summary,
                        image_analysis_result=response.image_analysis_result,
                    )
                )

            except ValidationError as exc:
                st.warning(f"⚠️ {exc.message}")
                logger.warning(f"Validation error for user_id={user_id}: {exc}")
            except (AgentExecutionError, StyleSenseBaseError) as exc:
                error_msg = "I hit a snag processing that. Could you try rephrasing your request?"
                st.error(error_msg)
                logger.error(f"Agent error for user_id={user_id}: {exc}")
                st.session_state.chat_messages.append(
                    ChatTurn(user_text=user_input, assistant_text=error_msg)
                )
