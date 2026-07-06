"""
Review summarization tool.

Loads reviews (from sample data by default, or optionally live-scraped via
scraper_service) and produces a structured pros/cons summary using Gemini.
"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.tools import tool

from src.prompts.review_summary_prompts import build_review_summary_prompt
from src.services.gemini_service import get_gemini_service
from src.services.scraper_service import ScrapedReview, get_scraper_service
from src.utils.exceptions import GeminiAPIError, ProductCatalogError, ScraperError
from src.utils.logger import get_logger

logger = get_logger(__name__)

_SAMPLE_REVIEWS_PATH = Path("data/sample_reviews.json")


def load_sample_reviews(product_id: str) -> list[str]:
    """
    Load sample review texts for a given product from the bundled sample
    dataset (used as the default/offline data source).

    Args:
        product_id: The product ID to fetch reviews for.

    Returns:
        List of review text strings. Empty list if none found.

    Raises:
        ProductCatalogError: If the sample reviews file is missing or malformed.
    """
    if not _SAMPLE_REVIEWS_PATH.exists():
        raise ProductCatalogError(
            f"Sample reviews file not found at '{_SAMPLE_REVIEWS_PATH}'."
        )

    try:
        all_reviews: dict = json.loads(_SAMPLE_REVIEWS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProductCatalogError("Sample reviews file is malformed.", details=str(exc)) from exc

    product_reviews = all_reviews.get(product_id, [])
    return [r.get("text", "") for r in product_reviews if r.get("text")]


def scrape_live_reviews(source_url: str) -> list[str]:
    """
    Attempt to scrape live reviews from a product's source page.

    Args:
        source_url: The URL of the product's review page.

    Returns:
        List of scraped review text strings.

    Raises:
        ScraperError: If the page cannot be fetched or parsed.
    """
    scraper = get_scraper_service()
    soup = scraper.fetch_page(source_url)
    scraped: list[ScrapedReview] = scraper.extract_reviews(soup)
    return [r.text for r in scraped if r.text]


def summarize_reviews(
    product_name: str,
    product_id: str,
    *,
    source_url: str | None = None,
    prefer_live_scraping: bool = False,
) -> dict:
    """
    Summarize reviews for a product into structured pros/cons/sentiment.

    Args:
        product_name: Display name of the product (for the prompt).
        product_id: Product ID used to look up sample reviews.
        source_url: Optional live review page URL to scrape.
        prefer_live_scraping: If True and source_url is set, attempt live
            scraping first, falling back to sample data on failure.

    Returns:
        Structured summary dictionary with overall_sentiment, pros, cons,
        fit_notes, quality_notes, and summary.
    """
    review_texts: list[str] = []

    if prefer_live_scraping and source_url:
        try:
            review_texts = scrape_live_reviews(source_url)
            logger.info(f"Scraped {len(review_texts)} live reviews for '{product_name}'.")
        except ScraperError as exc:
            logger.warning(f"Live scraping failed, falling back to sample data: {exc}")

    if not review_texts:
        try:
            review_texts = load_sample_reviews(product_id)
        except ProductCatalogError as exc:
            logger.warning(f"No sample reviews available for '{product_id}': {exc}")

    if not review_texts:
        return {
            "overall_sentiment": "unknown",
            "pros": [],
            "cons": [],
            "fit_notes": None,
            "quality_notes": None,
            "summary": f"No reviews are currently available for {product_name}.",
        }

    prompt = build_review_summary_prompt(product_name, review_texts)

    try:
        gemini = get_gemini_service()
        return gemini.generate_structured_json(prompt, temperature=0.3)
    except GeminiAPIError as exc:
        logger.error(f"Review summarization failed for '{product_name}': {exc}")
        return {
            "overall_sentiment": "unknown",
            "pros": [],
            "cons": [],
            "fit_notes": None,
            "quality_notes": None,
            "summary": "Review summary is temporarily unavailable.",
        }


@tool
def review_summarizer_tool_fn(product_id: str, product_name: str) -> dict:
    """
    LangChain tool interface: summarize reviews for a given product into
    structured pros/cons and overall sentiment.

    Args:
        product_id: The product ID to summarize reviews for.
        product_name: Display name of the product.

    Returns:
        Structured review summary dictionary.
    """
    return summarize_reviews(product_name, product_id)
