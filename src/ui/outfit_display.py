"""
Streamlit outfit and result rendering components.

Renders Outfit recommendations, product comparisons, and review summaries
as structured, visually organized cards rather than plain text blobs.
"""

from __future__ import annotations

import streamlit as st

from src.models.outfit import Outfit


def render_outfit_card(outfit: Outfit) -> None:
    """
    Render a complete Outfit recommendation as a structured card with
    item details, total price, style summary, reasoning, and color harmony.

    Args:
        outfit: The Outfit object to display.
    """
    with st.container(border=True):
        st.markdown(f"### 🛍️ {outfit.style_summary or 'Your Outfit Recommendation'}")

        cols = st.columns(min(len(outfit.items), 4) or 1)
        for idx, item in enumerate(outfit.items):
            with cols[idx % len(cols)]:
                if item.image_url:
                    st.image(item.image_url, use_container_width=True)
                st.markdown(f"**{item.name}**")
                st.caption(f"{item.brand} · ${item.price:.2f}")
                if item.colors:
                    st.caption(f"Colors: {', '.join(item.colors)}")

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Price", f"${outfit.total_price:.2f}")
        with col2:
            if outfit.occasion:
                st.metric("Occasion", outfit.occasion.title())

        if outfit.reasoning:
            st.markdown(f"**Why this works:** {outfit.reasoning}")

        if outfit.color_harmony:
            st.markdown(
                f"**Color harmony:** {outfit.color_harmony.harmony_type.title()} "
                f"({outfit.color_harmony.score:.0%} match) — {outfit.color_harmony.explanation}"
            )

        if outfit.alternatives:
            with st.expander(f"See {len(outfit.alternatives)} alternative option(s)"):
                for alt in outfit.alternatives:
                    render_outfit_card(alt)


def render_comparison_result(comparison: dict) -> None:
    """
    Render a structured product comparison result as a table with a
    highlighted recommendation.

    Args:
        comparison: Structured comparison dict from comparison_tool.
    """
    with st.container(border=True):
        st.markdown("### ⚖️ Product Comparison")

        table_data = comparison.get("comparison_table", [])
        if table_data:
            st.table(
                [
                    {
                        "Product ID": row.get("product_id", ""),
                        "Value": row.get("value_for_money", ""),
                        "Style Fit": row.get("style_fit", ""),
                        "Strength": row.get("notable_strength", ""),
                        "Weakness": row.get("notable_weakness", "—"),
                    }
                    for row in table_data
                ]
            )

        recommendation = comparison.get("recommendation")
        if recommendation:
            st.success(f"**Recommended: {recommendation}**")
            st.caption(comparison.get("recommendation_reasoning", ""))


def render_review_summary(summary: dict) -> None:
    """
    Render a structured review summary as pros/cons columns with an
    overall sentiment badge.

    Args:
        summary: Structured review summary dict from review_summarizer_tool.
    """
    with st.container(border=True):
        st.markdown("### 📝 Review Summary")

        sentiment = summary.get("overall_sentiment", "unknown")
        sentiment_emoji = {"positive": "😊", "mixed": "😐", "negative": "😕"}.get(sentiment, "❓")
        st.markdown(f"**Overall sentiment:** {sentiment_emoji} {sentiment.title()}")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**👍 Pros**")
            for pro in summary.get("pros", []):
                st.markdown(f"- {pro}")
        with col2:
            st.markdown("**👎 Cons**")
            for con in summary.get("cons", []):
                st.markdown(f"- {con}")

        if summary.get("fit_notes"):
            st.caption(f"**Fit notes:** {summary['fit_notes']}")
        if summary.get("quality_notes"):
            st.caption(f"**Quality notes:** {summary['quality_notes']}")

        st.markdown(summary.get("summary", ""))


def render_image_analysis_result(analysis: dict) -> None:
    """
    Render structured image analysis results as a compact info panel.

    Args:
        analysis: Structured analysis dict from image_analysis_tool.
    """
    with st.container(border=True):
        st.markdown("### 🔍 Image Analysis")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Type:** {analysis.get('garment_type', 'unknown').title()}")
            st.markdown(f"**Pattern:** {analysis.get('pattern', 'N/A')}")
            st.markdown(f"**Material guess:** {analysis.get('material_guess', 'N/A')}")
        with col2:
            colors = analysis.get("dominant_colors", [])
            st.markdown(f"**Dominant colors:** {', '.join(colors) if colors else 'N/A'}")
            styles = analysis.get("style_tags", [])
            st.markdown(f"**Style:** {', '.join(styles) if styles else 'N/A'}")

        occasions = analysis.get("suitable_occasions", [])
        if occasions:
            st.caption(f"Suitable for: {', '.join(occasions)}")