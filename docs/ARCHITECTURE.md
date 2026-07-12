# StyleSense AI – System Architecture

## 1. Overview

StyleSense AI is built around a **single LangGraph-orchestrated agent** that reasons over a shared state object, calling specialized tools as needed and persisting context via a conversation memory layer. This document describes the architecture, data flow, and key design decisions.

---

## 2. High-Level Architecture Diagram

```
                         ┌─────────────────────────────┐
                         │         Streamlit UI          │
                         │  (chat_interface, sidebar,     │
                         │   image_uploader, outfit_display)│
                         └───────────────┬─────────────┘
                                         │ user message + optional image
                                         ▼
                         ┌─────────────────────────────┐
                         │      Shopping Agent (LangGraph) │
                         │   src/agents/shopping_agent.py  │
                         └───────────────┬─────────────┘
                                         │
                 ┌───────────────────────┼───────────────────────┐
                 ▼                       ▼                       ▼
        ┌─────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐
        │  Intent Router   │   │  Conversation Memory  │   │   Tool Dispatcher    │
        │  (graph node)    │   │  (Chroma/FAISS +       │   │ (graph node routing)  │
        │                  │   │  chat history buffer)  │   │                       │
        └─────────────────┘   └─────────────────────┘   └──────────┬────────────┘
                                                                     │
       ┌─────────────────────────┬─────────────────┬────────────────┼──────────────────┬────────────────────┐
       ▼                         ▼                 ▼                ▼                  ▼                    ▼
┌─────────────┐         ┌───────────────┐  ┌───────────────┐ ┌───────────────┐ ┌────────────────┐  ┌─────────────────┐
│ Image        │         │ Product Search │  │ Color Matching │ │ Budget Filter  │ │ Weather/Occasion │  │ Review Summarizer│
│ Analysis Tool│         │ Tool           │  │ Tool           │ │ Tool           │ │ Tool             │  │ / Comparison Tool│
└──────┬───────┘         └───────┬───────┘  └───────┬───────┘ └───────┬───────┘ └────────┬─────────┘  └────────┬─────────┘
       │                         │                   │                 │                  │                     │
       ▼                         ▼                   ▼                 ▼                  ▼                     ▼
┌─────────────┐         ┌───────────────────────────────────────────────────────────────────────────────────────┐
│ Gemini       │         │                     Services Layer (gemini_service, scraper_service,                  │
│ Vision API   │         │                     weather_service, product_catalog_service)                        │
└─────────────┘         └───────────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
                         ┌─────────────────────────────┐
                         │   Bundle Generator Tool       │
                         │   → Final Outfit Response      │
                         └─────────────────────────────┘
```

---

## 3. Agent Design: LangGraph State Machine

### 3.1 State Schema (`src/agents/agent_state.py`)

The agent's state is a `TypedDict` threaded through every node:

```python
class AgentState(TypedDict):
    user_message: str
    uploaded_image: Optional[bytes]
    conversation_history: List[Message]
    detected_intent: Optional[str]
    extracted_entities: Dict[str, Any]   # budget, occasion, colors, weather, etc.
    image_analysis_result: Optional[Dict[str, Any]]
    retrieved_products: List[Product]
    recommended_outfit: Optional[Outfit]
    final_response: Optional[str]
    error: Optional[str]
```

### 3.2 Graph Nodes

| Node | Responsibility |
|---|---|
| `intent_router` | Classifies user intent (recommend, compare, summarize_reviews, general_chat) |
| `image_analysis_node` | Invokes Gemini Vision if an image is present |
| `entity_extraction_node` | Extracts budget, occasion, weather, color preferences from text |
| `retrieval_node` | Queries vector store / product catalog for candidates |
| `tool_dispatch_node` | Routes to color/budget/weather/comparison/review tools based on intent |
| `bundle_assembly_node` | Assembles final outfit bundle with reasoning |
| `response_formatter_node` | Formats final natural-language response |

### 3.3 Edges & Conditional Routing

The graph uses **conditional edges** keyed on `detected_intent`:

```
intent_router
   ├── "recommend_outfit"   → entity_extraction_node → retrieval_node → bundle_assembly_node
   ├── "compare_products"   → comparison_tool_node
   ├── "summarize_reviews"  → review_summarizer_node
   ├── "analyze_image"      → image_analysis_node → entity_extraction_node → ...
   └── "general_chat"       → response_formatter_node (direct LLM reply)
```

This keeps latency low for simple chat turns while allowing full tool orchestration for complex requests.

---

## 4. Memory Architecture

Two complementary memory systems:

1. **Conversation Memory** (`src/memory/conversation_memory.py`)
   Short-term buffer of the last N turns (configurable via `MAX_CONVERSATION_HISTORY`), passed directly into the LLM context window.

2. **Vector Memory / Product Index** (`src/memory/vector_store.py`)
   ChromaDB (default) or FAISS index over the product catalog, embedded once at startup. Also used to store long-term user preference embeddings (`user_profile_memory.py`) for personalization across sessions.

---

## 5. Multi-Modal Image Analysis Flow

1. User uploads an image via `src/ui/image_uploader.py`.
2. Image is validated (size/type) via `src/utils/validators.py` and preprocessed via `src/utils/image_utils.py` (resize, format conversion, base64 encoding).
3. `image_analysis_tool.py` sends the image + a structured prompt (`prompts/image_analysis_prompts.py`) to Gemini Vision via `gemini_service.py`.
4. The structured JSON result (detected garment type, dominant colors, style tags) is merged into `AgentState.image_analysis_result` and used downstream by color matching and retrieval nodes.

---

## 6. Error Handling & Resilience

- All external API calls (Gemini, weather, scraping) are wrapped with `tenacity` retry decorators (exponential backoff, max 3 attempts).
- Custom exceptions in `src/utils/exceptions.py` (`GeminiAPIError`, `ScraperError`, `InvalidImageError`, `BudgetConstraintError`) allow the agent to gracefully degrade — e.g., falling back to cached/sample data if scraping fails.
- All errors are logged via `src/utils/logger.py` (loguru-based, rotating file + console handlers) and surfaced to the user as friendly messages, never raw stack traces.

---

## 7. Design Decisions & Trade-offs

| Decision | Rationale |
|---|---|
| LangGraph over plain LangChain chains | Explicit state machine makes conditional tool routing and debugging far easier than nested chains |
| ChromaDB default over FAISS | Simpler local persistence, no manual index management; FAISS offered as a swap-in for scale |
| Sample JSON catalog instead of live scraping by default | Keeps the repo runnable offline/without rate-limit risk; scraper service is real but opt-in |
| Structured JSON outputs from Gemini prompts | Enables deterministic downstream parsing instead of fragile string parsing |

---

## 8. Sequence Diagram: "Recommend an outfit for a beach wedding, $150 budget"

```
User → UI: sends message
UI → Agent: invoke(state)
Agent → intent_router: classify → "recommend_outfit"
Agent → entity_extraction_node: {occasion: "beach wedding", budget: 150}
Agent → retrieval_node: query vector store (occasion + style filters)
retrieval_node → product_catalog_service: fetch candidates
Agent → tool_dispatch: color_matching_tool + budget_filter_tool
Agent → bundle_assembly_node: assemble top/bottom/shoes/accessories
Agent → response_formatter_node: generate explanation
Agent → UI: final_response + outfit cards
UI → User: renders chat bubble + outfit_display cards
```