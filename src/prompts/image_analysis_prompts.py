"""
Prompt templates for Gemini Vision image analysis tasks.
"""

from __future__ import annotations

GARMENT_ANALYSIS_PROMPT = """\
You are a fashion expert analyzing a photo of a clothing item or outfit. \
Examine the image carefully and extract structured details.

Respond with ONLY valid JSON in this exact format, no markdown fences:
{
  "garment_type": "<e.g. top, bottom, dress, outerwear, shoes, accessory>",
  "dominant_colors": ["<color>", ...],
  "secondary_colors": ["<color>", ...],
  "style_tags": ["<e.g. casual, formal, streetwear, minimalist, bohemian>"],
  "pattern": "<e.g. solid, striped, floral, plaid, none>",
  "material_guess": "<best guess, e.g. cotton, denim, leather, silk>",
  "fit_description": "<e.g. slim fit, oversized, tailored>",
  "suitable_occasions": ["<occasion>", ...],
  "suitable_seasons": ["<season>", ...],
  "confidence": <float 0.0-1.0>
}

If the image does not clearly show a clothing item, set garment_type to \
"unclear" and explain nothing further — just return the JSON with your \
best-effort guesses and a low confidence score.
"""


OUTFIT_MATCHING_FROM_IMAGE_PROMPT = """\
You are a fashion stylist. A user uploaded a photo of a clothing item shown \
in the image, described as: {garment_description}

They want recommendations for what to pair with it. Consider color \
harmony, style consistency, and versatility.

Respond with ONLY valid JSON in this exact format, no markdown fences:
{{
  "recommended_categories_to_pair": ["<category>", ...],
  "recommended_colors": ["<color>", ...],
  "styling_notes": "<2-3 sentences of styling advice for pairing with this item>"
}}
"""


IMAGE_QUALITY_CHECK_PROMPT = """\
Examine this image and determine if it is suitable for fashion/clothing \
analysis (clear, well-lit, shows a garment or outfit clearly).

Respond with ONLY valid JSON in this exact format, no markdown fences:
{
  "is_suitable": <true or false>,
  "issue": "<e.g. 'too blurry', 'no garment visible', 'too dark', or null if suitable>"
}
"""
