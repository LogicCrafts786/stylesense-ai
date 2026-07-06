# StyleSense AI – Prompt Engineering Notes

This document explains the reasoning behind the prompt design choices in
`src/prompts/`, for contributors extending or tuning agent behavior.

---

## 1. Design Principles

### Structured JSON output everywhere
Every prompt used for anything other than the final user-facing response
requests **strict JSON output with no markdown fences or preamble**. This
is deliberate: parsing free-form LLM text is fragile, while JSON parsing
is deterministic and lets `gemini_service.generate_structured_json()`
fail loudly and specifically (`GeminiAPIError`) rather than silently
misinterpreting text.

Example instruction pattern used throughout:
```
Respond with ONLY valid JSON in this exact format, with no markdown
fences or additional text: { ... }
```

### Two-stage generation: structured reasoning → natural language
The graph never shows raw JSON to the user. Nodes like
`bundle_assembly_node` produce structured `Outfit` objects, and a
separate `response_formatter_node` call turns that structured data into
warm, conversational text via the `STYLIST_PERSONA_SYSTEM_PROMPT`. This
separation means:
- Structured data stays reliable for downstream UI rendering (outfit
  cards, comparison tables).
- Tone/persona can be iterated on independently without touching
  extraction logic.

### Low temperature for extraction, higher for creative framing
- Intent classification, entity extraction, image analysis: `temperature=0.1–0.3`
  (deterministic, repeatable outputs).
- Outfit reasoning, final response phrasing: `temperature=0.5–0.7`
  (some creative variation is desirable in styling language).

### Never let the LLM invent products
All product-selection prompts (`build_outfit_reasoning_prompt`,
`build_alternative_outfit_prompt`) explicitly instruct: *"Only choose
from the candidate list below — do not invent items."* The candidate
list itself is pre-filtered by deterministic Python code (budget,
category, occasion) before ever reaching Gemini — the LLM's job is
narrowing/reasoning over real inventory, not generating catalog data.

---

## 2. Prompt Catalog

| Prompt | File | Purpose | Temp |
|---|---|---|---|
| `STYLIST_PERSONA_SYSTEM_PROMPT` | `system_prompts.py` | Persona/tone for all user-facing responses | 0.7 |
| `INTENT_CLASSIFICATION_SYSTEM_PROMPT` | `system_prompts.py` | Classify user intent into 5 categories | 0.1 |
| `ENTITY_EXTRACTION_SYSTEM_PROMPT` | `system_prompts.py` | Extract budget/occasion/colors/styles | 0.1 |
| `build_outfit_reasoning_prompt()` | `outfit_prompts.py` | Select + justify outfit from candidates | 0.5 |
| `build_color_matching_prompt()` | `outfit_prompts.py` | Color theory analysis of a palette | 0.3 |
| `GARMENT_ANALYSIS_PROMPT` | `image_analysis_prompts.py` | Extract structured attrs from an image | 0.2 |
| `build_review_summary_prompt()` | `review_summary_prompts.py` | Pros/cons summary from raw reviews | 0.3 |
| `build_product_comparison_prompt()` | `review_summary_prompts.py` | Structured product comparison | 0.3 |

---

## 3. Fallback Strategy

Every LLM-dependent tool has a **non-LLM fallback path**:

- `color_matching_tool.analyze_color_harmony()` falls back to a
  rule-based color-wheel heuristic (`_rule_based_harmony_score`) if
  Gemini fails or `use_llm_refinement=False`.
- `bundle_generator_tool.generate_outfit_bundle()` falls back to the
  first 3 candidate products if Gemini returns no valid product IDs.
- `weather_tool` works entirely from static JSON mapping
  (`occasion_weather_map.json`) if no live weather API key is configured.

This ensures the app degrades gracefully rather than hard-failing when
the LLM is unavailable, rate-limited, or returns malformed output.

---

## 4. Extending Prompts

When adding a new tool/prompt:
1. Always specify the exact JSON schema expected, field by field.
2. State explicitly what to do when a field is unknown (`null`, not omission).
3. Keep instructions declarative ("respond with X") rather than
   conversational ("could you please respond with X") — this measurably
   improves JSON-only compliance.
4. Add a rule-based or cached fallback wherever feasible.
5. Document the new prompt in the table above.
