"""
Streamlit sidebar UI components.

Renders user profile info, budget/occasion quick-set controls, and
session management actions (reset conversation, view preferences).
"""

from __future__ import annotations

import streamlit as st

from src.agents.shopping_agent import get_shopping_agent
from src.memory.user_profile_memory import get_user_profile_memory
from src.utils.logger import get_logger

logger = get_logger(__name__)


def render_sidebar(user_id: str) -> None:
    """
    Render the full sidebar: app branding, user preference summary, quick
    context controls, and session reset action.

    Args:
        user_id: The current session's user identifier.
    """
    with st.sidebar:
        st.markdown("## 👗 StyleSense AI")
        st.caption("Your AI-powered personal shopping stylist")
        st.divider()

        _render_preference_summary(user_id)
        st.divider()

        _render_quick_context_controls(user_id)
        st.divider()

        _render_session_controls(user_id)


def _render_preference_summary(user_id: str) -> None:
    """
    Display a read-only summary of the user's known long-term preferences.

    Args:
        user_id: The current session's user identifier.
    """
    st.markdown("### Your Style Profile")

    profile_memory = get_user_profile_memory()
    preferences = profile_memory.get_or_create(user_id)

    if preferences.preferred_colors:
        st.markdown(f"**Loved colors:** {', '.join(preferences.preferred_colors)}")
    else:
        st.caption("No color preferences learned yet — just start chatting!")

    if preferences.preferred_styles:
        st.markdown(f"**Style vibe:** {', '.join(preferences.preferred_styles)}")

    if preferences.favorite_brands:
        st.markdown(f"**Favorite brands:** {', '.join(preferences.favorite_brands)}")

    if preferences.current_occasion:
        st.info(f"📌 Current occasion: **{preferences.current_occasion}**")

    if preferences.current_budget:
        st.info(f"💰 Current budget: **${preferences.current_budget:.2f}**")


def _render_quick_context_controls(user_id: str) -> None:
    """
    Render quick-set widgets for occasion and budget, letting the user set
    context without typing it in chat. Saves directly to profile memory.

    Args:
        user_id: The current session's user identifier.
    """
    st.markdown("### Quick Context")

    profile_memory = get_user_profile_memory()
    preferences = profile_memory.get_or_create(user_id)

    occasion_input = st.text_input(
        "Occasion",
        value=preferences.current_occasion or "",
        placeholder="e.g. beach wedding, job interview",
        key="sidebar_occasion_input",
    )
    budget_input = st.number_input(
        "Budget ($)",
        min_value=0.0,
        value=float(preferences.current_budget or 0.0),
        step=10.0,
        key="sidebar_budget_input",
    )

    if st.button("Update Context", use_container_width=True):
        preferences.current_occasion = occasion_input.strip() or None
        preferences.current_budget = budget_input if budget_input > 0 else None
        profile_memory.save(preferences)
        st.success("Context updated!")
        logger.info(f"User '{user_id}' manually updated context via sidebar.")


def _render_session_controls(user_id: str) -> None:
    """
    Render session management actions: reset conversation history.

    Args:
        user_id: The current session's user identifier.
    """
    st.markdown("### Session")

    if st.button("🔄 Start New Conversation", use_container_width=True):
        agent = get_shopping_agent()
        agent.reset_conversation(user_id)
        st.session_state.chat_messages = []
        st.success("Conversation reset!")
        st.rerun()

    st.caption("Resetting clears chat history but keeps your learned style preferences.")
