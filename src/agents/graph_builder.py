"""
LangGraph node and edge definitions for the StyleSense shopping agent.

Each node is a plain function: (AgentState) -> partial AgentState update.
Conditional routing is driven by `detected_intent` after the intent_router
node runs, keeping simple chat turns cheap and complex requests fully
orchestrated through the relevant tools.
"""

from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, StateGraph

from src.agents.agent_state import AgentState
from src.memory.user_profile_memory import get_user_profile_memory
from src.prompts.system_prompts import (
    ENTITY_EXTRACTION_SYSTEM_PROMPT,
    GENERAL_CHAT_SYSTEM_PROMPT,
    INTENT_CLASSIFICATION_SYSTEM_PROMPT,
    STYLIST_PERSONA_SYSTEM_PROMPT,
)
from src.services.gemini_service import get_gemini_service
from src.tools.bundle_generator_tool import generate_outfit_bundle
from src.tools.comparison_tool import compare_products
from src.tools.image_analysis_tool import analyze_garment_image
from src.tools.product_search_tool import get_products_by_ids, search_products
from src.tools.review_summarizer_tool import summarize_reviews
from src.tools.weather_tool import get_garment_guidance_for_context, get_weather_context
from src.utils.exceptions import (
    AgentExecutionError,
    BudgetConstraintError,
    GeminiAPIError,
    ImageAnalysisError,
    ProductCatalogError,
    ValidationError,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

_VALID_INTENTS = {
    "recommend_outfit",
    "compare_products",
    "summarize_reviews",
    "analyze_image",
    "general_chat",
}


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def intent_router_node(state: AgentState) -> dict[str, Any]:
    """
    Classify the user's intent for this turn using Gemini.

    Args:
        state: Current agent state.

    Returns:
        Partial state update with detected_intent and intent_confidence.
    """
    try:
        gemini = get_gemini_service()

        prompt = f"User message: {state['user_message']}\nImage attached: {bool(state.get('image_bytes'))}"
        result = gemini.generate_structured_json(
            prompt, system_instruction=INTENT_CLASSIFICATION_SYSTEM_PROMPT, temperature=0.1
        )

        intent = result.get("intent", "general_chat")
        if intent not in _VALID_INTENTS:
            logger.warning(f"Gemini returned unrecognized intent '{intent}', defaulting to general_chat.")
            intent = "general_chat"

        # If an image was uploaded, prioritize image analysis regardless of
        # text classification, since the image is likely central to the request.
        if state.get("image_bytes") and intent == "general_chat":
            intent = "analyze_image"

        logger.info(f"Detected intent: {intent} (confidence={result.get('confidence')})")
        return {
            "detected_intent": intent,
            "intent_confidence": float(result.get("confidence", 0.5)),
            "tool_calls_made": state.get("tool_calls_made", []) + ["intent_router"],
        }

    except GeminiAPIError as exc:
        logger.error(f"Intent classification failed: {exc}")
        return {"detected_intent": "general_chat", "error": None}


def entity_extraction_node(state: AgentState) -> dict[str, Any]:
    """
    Extract structured shopping entities (budget, occasion, colors, styles,
    weather) from the user's message.

    Args:
        state: Current agent state.

    Returns:
        Partial state update with extracted_entities populated.
    """
    try:
        gemini = get_gemini_service()
        prompt = f"User message: {state['user_message']}"
        entities = gemini.generate_structured_json(
            prompt, system_instruction=ENTITY_EXTRACTION_SYSTEM_PROMPT, temperature=0.1
        )

        # Persist relevant signals to long-term user profile memory
        profile_memory = get_user_profile_memory()
        sentiment_map = entities.get("sentiment_on_colors", {}) or {}
        colors_liked = [c for c, sentiment in sentiment_map.items() if sentiment == "liked"]
        colors_disliked = [c for c, sentiment in sentiment_map.items() if sentiment == "disliked"]

        profile_memory.update_from_entities(
            state["user_id"],
            colors_liked=colors_liked or entities.get("colors_mentioned", []),
            colors_disliked=colors_disliked,
            styles=entities.get("style_keywords", []),
            current_occasion=entities.get("occasion"),
            current_budget=entities.get("budget"),
            current_weather_context=entities.get("weather_or_season"),
        )

        logger.info(f"Extracted entities: {entities}")
        return {
            "extracted_entities": entities,
            "tool_calls_made": state.get("tool_calls_made", []) + ["entity_extraction"],
        }

    except GeminiAPIError as exc:
        logger.error(f"Entity extraction failed: {exc}")
        return {"extracted_entities": {}, "error": None}


def image_analysis_node(state: AgentState) -> dict[str, Any]:
    """
    Analyze an uploaded image via Gemini Vision, if present in the state.

    Args:
        state: Current agent state.

    Returns:
        Partial state update with image_analysis_result populated.
    """
    image_bytes = state.get("image_bytes")
    if not image_bytes:
        return {"image_analysis_result": None}

    try:
        result = analyze_garment_image(image_bytes)
        logger.info(f"Image analysis result: garment_type={result.get('garment_type')}")
        return {
            "image_analysis_result": result,
            "tool_calls_made": state.get("tool_calls_made", []) + ["image_analysis_tool"],
        }
    except ImageAnalysisError as exc:
        logger.error(f"Image analysis node failed: {exc}")
        return {"image_analysis_result": None, "error": f"I couldn't analyze that image: {exc.message}"}


def weather_node(state: AgentState) -> dict[str, Any]:
    """
    Retrieve weather context and derive garment guidance based on the
    extracted occasion and any location mentioned.

    Args:
        state: Current agent state.

    Returns:
        Partial state update with weather_context populated.
    """
    entities = state.get("extracted_entities", {}) or {}
    occasion = entities.get("occasion")
    weather_hint = entities.get("weather_or_season")

    # If the user didn't mention a location, we can't hit the live weather
    # API, but we can still apply season-hint-based garment guidance.
    guidance = get_garment_guidance_for_context(occasion=occasion, season_hint=weather_hint)

    return {
        "weather_context": guidance,
        "tool_calls_made": state.get("tool_calls_made", []) + ["weather_tool"],
    }


def retrieval_node(state: AgentState) -> dict[str, Any]:
    """
    Retrieve candidate products from the catalog based on extracted
    entities and any image analysis results.

    Args:
        state: Current agent state.

    Returns:
        Partial state update with candidate_products populated.
    """
    entities = state.get("extracted_entities", {}) or {}
    image_result = state.get("image_analysis_result") or {}

    categories = entities.get("product_categories_mentioned", [])
    if not categories and image_result.get("garment_type"):
        # If an image was analyzed, infer complementary categories to search
        categories = ["top", "bottom", "shoes", "accessory"]

    colors = entities.get("colors_mentioned", []) or image_result.get("dominant_colors", [])
    budget = entities.get("budget")
    occasion = entities.get("occasion")

    try:
        products = search_products(
            categories=categories or None,
            max_price=budget,
            colors=colors or None,
            occasion=occasion,
            limit=25,
        )
        logger.info(f"Retrieved {len(products)} candidate products.")
        return {
            "candidate_products": products,
            "tool_calls_made": state.get("tool_calls_made", []) + ["product_search_tool"],
        }
    except ProductCatalogError as exc:
        logger.error(f"Product retrieval failed: {exc}")
        return {"candidate_products": [], "error": f"I had trouble searching the catalog: {exc.message}"}


def bundle_assembly_node(state: AgentState) -> dict[str, Any]:
    """
    Assemble a cohesive outfit from candidate products using the
    bundle_generator_tool.

    Args:
        state: Current agent state.

    Returns:
        Partial state update with recommended_outfit populated.
    """
    candidates = state.get("candidate_products", [])
    entities = state.get("extracted_entities", {}) or {}
    weather_ctx = state.get("weather_context", {}) or {}

    if not candidates:
        return {
            "error": (
                "I couldn't find any products matching your criteria. "
                "Could you try adjusting your budget or being more specific about style?"
            )
        }

    profile_memory = get_user_profile_memory()
    preferences = profile_memory.get_or_create(state["user_id"])

    try:
        outfit = generate_outfit_bundle(
            candidate_products=candidates,
            occasion=entities.get("occasion") or preferences.current_occasion,
            budget=entities.get("budget") or preferences.current_budget,
            weather_description=weather_ctx.get("condition"),
            preferred_colors=preferences.preferred_colors,
            preferred_styles=preferences.preferred_styles,
        )
        return {
            "recommended_outfit": outfit,
            "tool_calls_made": state.get("tool_calls_made", []) + ["bundle_generator_tool"],
        }
    except (AgentExecutionError, BudgetConstraintError) as exc:
        logger.error(f"Bundle assembly failed: {exc}")
        return {"error": f"I ran into trouble putting together an outfit: {exc.message}"}


def comparison_node(state: AgentState) -> dict[str, Any]:
    """
    Handle product comparison requests by extracting referenced product IDs
    (via candidate retrieval) and generating a structured comparison.

    Args:
        state: Current agent state.

    Returns:
        Partial state update with comparison_result populated.
    """
    entities = state.get("extracted_entities", {}) or {}
    categories = entities.get("product_categories_mentioned", [])

    try:
        candidates = search_products(categories=categories or None, limit=5)
        if len(candidates) < 2:
            return {"error": "I need at least two specific products in mind to compare. Could you name them?"}

        result = compare_products(candidates[:4])
        return {
            "comparison_result": result,
            "tool_calls_made": state.get("tool_calls_made", []) + ["comparison_tool"],
        }
    except (ValidationError, GeminiAPIError) as exc:
        logger.error(f"Comparison node failed: {exc}")
        return {"error": f"I couldn't complete that comparison: {exc.message}"}


def review_summary_node(state: AgentState) -> dict[str, Any]:
    """
    Handle review summarization requests for a product mentioned by the user.

    Args:
        state: Current agent state.

    Returns:
        Partial state update with review_summary populated.
    """
    entities = state.get("extracted_entities", {}) or {}
    categories = entities.get("product_categories_mentioned", [])

    candidates = search_products(categories=categories or None, limit=1)
    if not candidates:
        return {"error": "I couldn't find that product to summarize reviews for. Could you specify it by name?"}

    target = candidates[0]
    summary = summarize_reviews(target.name, target.product_id)
    return {
        "review_summary": summary,
        "tool_calls_made": state.get("tool_calls_made", []) + ["review_summarizer_tool"],
    }


def response_formatter_node(state: AgentState) -> dict[str, Any]:
    """
    Compose the final natural-language response to the user based on
    whatever results are present in the state (outfit, comparison, review
    summary, or general chat).

    Args:
        state: Current agent state.

    Returns:
        Partial state update with final_response populated.
    """
    if state.get("error"):
        return {"final_response": state["error"]}

    gemini = get_gemini_service()
    intent = state.get("detected_intent")

    try:
        if intent == "recommend_outfit" and state.get("recommended_outfit"):
            outfit = state["recommended_outfit"]
            item_lines = "\n".join(f"- {item.name} (${item.price:.2f}) in {', '.join(item.colors)}" for item in outfit.items)
            prompt = (
                f"Present this outfit recommendation to the user warmly and concisely.\n\n"
                f"Items:\n{item_lines}\n\nTotal: ${outfit.total_price:.2f}\n"
                f"Style summary: {outfit.style_summary}\nReasoning: {outfit.reasoning}\n"
                f"Color harmony: {outfit.color_harmony.explanation if outfit.color_harmony else 'N/A'}"
            )
            response = gemini.generate_text(prompt, system_instruction=STYLIST_PERSONA_SYSTEM_PROMPT)

        elif intent == "compare_products" and state.get("comparison_result"):
            comparison = state["comparison_result"]
            prompt = f"Present this product comparison result conversationally: {comparison}"
            response = gemini.generate_text(prompt, system_instruction=STYLIST_PERSONA_SYSTEM_PROMPT)

        elif intent == "summarize_reviews" and state.get("review_summary"):
            summary = state["review_summary"]
            prompt = f"Present this review summary conversationally: {summary}"
            response = gemini.generate_text(prompt, system_instruction=STYLIST_PERSONA_SYSTEM_PROMPT)

        elif intent == "analyze_image" and state.get("image_analysis_result"):
            analysis = state["image_analysis_result"]
            prompt = (
                f"The user uploaded an image of their item. Analysis: {analysis}\n"
                f"Their message: {state['user_message']}\n"
                "Describe what you see and offer styling suggestions."
            )
            response = gemini.generate_text(prompt, system_instruction=STYLIST_PERSONA_SYSTEM_PROMPT)

        else:
            response = gemini.generate_text(
                state["user_message"], system_instruction=GENERAL_CHAT_SYSTEM_PROMPT
            )

        return {"final_response": response}

    except GeminiAPIError as exc:
        logger.error(f"Response formatting failed: {exc}")
        return {
            "final_response": (
                "I've got the information ready, but I'm having trouble phrasing it right now. "
                "Please try asking again in a moment."
            )
        }


# ---------------------------------------------------------------------------
# Conditional routing
# ---------------------------------------------------------------------------

def route_after_intent(state: AgentState) -> Literal[
    "image_analysis", "entity_extraction", "comparison", "review_summary", "general_chat"
]:
    """
    Determine which path to take after intent classification.

    Args:
        state: Current agent state.

    Returns:
        The name of the next node/branch to execute.
    """
    intent = state.get("detected_intent", "general_chat")

    if intent == "analyze_image":
        return "image_analysis"
    if intent == "compare_products":
        return "comparison"
    if intent == "summarize_reviews":
        return "review_summary"
    if intent == "recommend_outfit":
        return "entity_extraction"
    return "general_chat"


def route_after_image_analysis(state: AgentState) -> Literal["entity_extraction", "response_formatter"]:
    """
    After image analysis, decide whether to continue into full outfit
    retrieval (if the user also wants recommendations) or go straight to
    formatting a description-only response.

    Args:
        state: Current agent state.

    Returns:
        The next node name.
    """
    message_lower = state["user_message"].lower()
    wants_recommendations = any(
        kw in message_lower for kw in ["pair", "match", "wear with", "recommend", "outfit", "goes with"]
    )
    return "entity_extraction" if wants_recommendations else "response_formatter"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_shopping_agent_graph() -> StateGraph:
    """
    Construct and compile the LangGraph state machine for the shopping agent.

    Returns:
        A compiled LangGraph application ready to invoke with an AgentState.
    """
    graph = StateGraph(AgentState)

    graph.add_node("intent_router", intent_router_node)
    graph.add_node("image_analysis", image_analysis_node)
    graph.add_node("entity_extraction", entity_extraction_node)
    graph.add_node("weather", weather_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("bundle_assembly", bundle_assembly_node)
    graph.add_node("comparison", comparison_node)
    graph.add_node("review_summary", review_summary_node)
    graph.add_node("response_formatter", response_formatter_node)

    graph.set_entry_point("intent_router")

    graph.add_conditional_edges(
        "intent_router",
        route_after_intent,
        {
            "image_analysis": "image_analysis",
            "comparison": "comparison",
            "review_summary": "review_summary",
            "entity_extraction": "entity_extraction",
            "general_chat": "response_formatter",
        },
    )

    graph.add_conditional_edges(
        "image_analysis",
        route_after_image_analysis,
        {
            "entity_extraction": "entity_extraction",
            "response_formatter": "response_formatter",
        },
    )

    graph.add_edge("entity_extraction", "weather")
    graph.add_edge("weather", "retrieval")
    graph.add_edge("retrieval", "bundle_assembly")
    graph.add_edge("bundle_assembly", "response_formatter")
    graph.add_edge("comparison", "response_formatter")
    graph.add_edge("review_summary", "response_formatter")
    graph.add_edge("response_formatter", END)

    return graph.compile()
