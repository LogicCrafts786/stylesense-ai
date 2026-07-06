# StyleSense AI – Internal API Reference

This document describes the primary internal module interfaces (not a
public REST API — StyleSense AI is a Streamlit application, not a
standalone API server).

---

## `src.agents.shopping_agent`

### `class ShoppingAgent`

The main orchestrator the UI layer talks to.

```python
from src.agents.shopping_agent import get_shopping_agent

agent = get_shopping_agent()
response = agent.handle_message(
    user_id="session-abc123",
    user_message="I need an outfit for a beach wedding, budget $150",
    image_bytes=None,  # or raw bytes of an uploaded image
)
```

**`handle_message(*, user_id, user_message, image_bytes=None) -> AgentResponse`**

| Param | Type | Description |
|---|---|---|
| `user_id` | `str` | Session/user identifier for memory scoping |
| `user_message` | `str` | Raw user chat input |
| `image_bytes` | `bytes \| None` | Optional uploaded image bytes |

Returns an `AgentResponse` dataclass with fields: `text`, `outfit`,
`comparison_result`, `review_summary`, `image_analysis_result`, `detected_intent`.

**`reset_conversation(user_id: str) -> None`**
Clears conversation history for a session while preserving long-term
user profile preferences (colors, styles, favorite brands).

---

## `src.services.gemini_service`

### `class GeminiService`

```python
from src.services.gemini_service import get_gemini_service

gemini = get_gemini_service()
text = gemini.generate_text("Suggest a color palette for autumn.")
data = gemini.generate_structured_json(prompt, system_instruction="...")
```

| Method | Purpose |
|---|---|
| `generate_text(prompt, *, system_instruction=None, temperature=0.7, max_output_tokens=1024)` | Plain text generation |
| `generate_structured_json(prompt, *, system_instruction=None, temperature=0.3)` | JSON-parsed structured generation |
| `analyze_image(encoded_image, prompt, *, temperature=0.2)` | Vision text response |
| `analyze_image_structured(encoded_image, prompt, *, temperature=0.2)` | Vision JSON-parsed response |

All methods raise `GeminiAPIError` on failure after retries.

---

## `src.services.product_catalog_service`

### `class ProductCatalogService`

```python
from src.services.product_catalog_service import get_product_catalog_service

catalog = get_product_catalog_service()
products = catalog.filter_products(max_price=100, occasion="beach wedding")
```

| Method | Purpose |
|---|---|
| `get_all_products() -> list[Product]` | Return entire catalog |
| `get_by_id(product_id) -> Product \| None` | Single product lookup |
| `filter_products(**kwargs) -> list[Product]` | Multi-criteria filtering |
| `search_by_predicate(fn) -> list[Product]` | Custom filter function |

---

## `src.tools.bundle_generator_tool`

```python
from src.tools.bundle_generator_tool import generate_outfit_bundle

outfit = generate_outfit_bundle(
    candidate_products=products,
    occasion="beach wedding",
    budget=150.0,
    weather_description="warm, sunny",
    preferred_colors=["white", "terracotta"],
    preferred_styles=["elegant"],
)
```

Returns an `Outfit` object (see `src.models.outfit`). Raises
`AgentExecutionError` if no candidates are provided or generation fails.

---

## `src.memory.conversation_memory`

```python
from src.memory.conversation_memory import get_conversation_memory_store

store = get_conversation_memory_store()
memory = store.get_or_create("session-abc123")
memory.add_user_message("Hi there")
context_text = memory.get_context_as_text()
```

---

## `src.memory.user_profile_memory`

```python
from src.memory.user_profile_memory import get_user_profile_memory

profile_memory = get_user_profile_memory()
prefs = profile_memory.get_or_create("session-abc123")
profile_memory.update_from_entities(
    "session-abc123", colors_liked=["navy"], current_budget=150.0
)
```

---

## `src.models`

| Model | Key Fields |
|---|---|
| `Product` | `product_id`, `name`, `category`, `price`, `colors`, `style_tags`, `occasion_tags` |
| `Outfit` | `outfit_id`, `items`, `total_price` (computed), `reasoning`, `color_harmony` |
| `UserPreferences` | `preferred_colors`, `preferred_styles`, `current_budget`, `current_occasion` |
| `Conversation` / `Message` | `messages`, `role`, `content`, `timestamp` |

All models expose `.to_dict()` / `.from_dict()` for JSON serialization.

---

## Exception Hierarchy

All custom exceptions extend `StyleSenseBaseError` (see `src/utils/exceptions.py`):

```
StyleSenseBaseError
├── ConfigurationError
├── GeminiAPIError
├── ImageAnalysisError
├── InvalidImageError
├── ScraperError
├── WeatherServiceError
├── ProductCatalogError
├── BudgetConstraintError
├── VectorStoreError
├── MemoryError_
├── AgentExecutionError
└── ValidationError
```

Each carries a `.message` and optional `.details` attribute for logging
vs. user-facing display.
