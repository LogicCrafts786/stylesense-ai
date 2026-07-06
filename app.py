"""
StyleSense AI — Main Streamlit Application Entry Point.

This module wires together configuration loading, session management,
and the UI layer (sidebar + chat interface) into a single runnable
Streamlit app. Run with:

    streamlit run app.py
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import streamlit as st

# Ensure the project root is on sys.path so `src.*` imports resolve
# correctly regardless of the working directory Streamlit is launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.ui.chat_interface import render_chat_interface  # noqa: E402
from src.ui.sidebar import render_sidebar  # noqa: E402
from src.utils.config import get_settings  # noqa: E402
from src.utils.exceptions import ConfigurationError  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

st.set_page_config(
    page_title="StyleSense AI — Personal Shopping Stylist",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _initialize_session() -> str:
    """
    Ensure a stable, unique user/session identifier exists for this browser
    session, used to scope conversation memory and user profile data.

    Returns:
        The session's unique user_id string.
    """
    if "user_id" not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())
        logger.info(f"New Streamlit session started: user_id={st.session_state.user_id}")
    return st.session_state.user_id


def _check_configuration() -> bool:
    """
    Validate that required configuration (e.g., GEMINI_API_KEY) is present
    before rendering the main app. Shows a friendly setup message if not.

    Returns:
        True if configuration is valid and the app should proceed,
        False if a blocking configuration error was displayed instead.
    """
    try:
        settings = get_settings()
        settings.validate_required_for_runtime()
        return True
    except ConfigurationError as exc:
        st.error("⚠️ Configuration Error")
        st.markdown(
            f"""
            **{exc.message}**

            {exc.details or ''}

            **To fix this:**
            1. Copy `.env.example` to `.env` in the project root.
            2. Add your Gemini API key: `GEMINI_API_KEY=your_key_here`
            3. Get a key at https://ai.google.dev/ if you don't have one.
            4. Restart the app.
            """
        )
        logger.error(f"Blocking configuration error: {exc}")
        return False


def _render_header() -> None:
    """Render the main page header and a short intro description."""
    st.title("👗 StyleSense AI")
    st.caption(
        "Your multi-modal AI shopping stylist — tell me about an occasion, "
        "budget, or upload a photo, and I'll help you build the perfect look."
    )


def main() -> None:
    """Application entry point: configuration check, session init, and UI render."""
    if not _check_configuration():
        st.stop()

    user_id = _initialize_session()

    _render_header()
    render_sidebar(user_id)
    render_chat_interface(user_id)


if __name__ == "__main__":
    main()
