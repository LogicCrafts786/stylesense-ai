"""
Web scraping service using Requests + BeautifulSoup.

Used primarily by the review_summarizer_tool to fetch product review
content from a product's source page. Includes polite rate limiting,
timeouts, and graceful degradation on scrape failures so the agent can
fall back to sample review data.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.config import get_settings
from src.utils.exceptions import ScraperError
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ScrapedReview:
    """
    Represents a single scraped review.

    Attributes:
        author: Reviewer display name, if available.
        rating: Numeric star rating (1-5), if parseable.
        text: The review body text.
    """

    author: str | None
    rating: float | None
    text: str


class ScraperService:
    """Fetches and parses HTML content from product/review pages."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": self._settings.scraper_user_agent})
        self._last_request_time: float = 0.0

    def _respect_rate_limit(self) -> None:
        """Sleep as needed to maintain a polite delay between requests."""
        elapsed = time.monotonic() - self._last_request_time
        delay = self._settings.scraper_rate_limit_delay
        if elapsed < delay:
            time.sleep(delay - elapsed)
        self._last_request_time = time.monotonic()

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        reraise=True,
    )
    def fetch_page(self, url: str) -> BeautifulSoup:
        """
        Fetch a web page and return a parsed BeautifulSoup object.

        Args:
            url: The target page URL.

        Returns:
            A BeautifulSoup object parsed from the page's HTML content.

        Raises:
            ScraperError: If the request fails, times out, or returns a
                non-success status code.
        """
        self._respect_rate_limit()
        try:
            response = self._session.get(url, timeout=self._settings.scraper_timeout_seconds)
            response.raise_for_status()
            return BeautifulSoup(response.text, "lxml")
        except requests.RequestException as exc:
            logger.error(f"Failed to fetch page '{url}': {exc}")
            raise ScraperError(f"Could not fetch content from '{url}'.", details=str(exc)) from exc

    def extract_reviews(
        self,
        soup: BeautifulSoup,
        *,
        review_container_selector: str = ".review",
        author_selector: str = ".review-author",
        rating_selector: str = ".review-rating",
        text_selector: str = ".review-text",
        max_reviews: int = 20,
    ) -> list[ScrapedReview]:
        """
        Extract review data from a parsed page using configurable CSS selectors.

        Note: Selector defaults are generic placeholders. In production,
        selectors should be tailored per retailer site structure, ideally
        loaded from a site-specific config rather than hardcoded.

        Args:
            soup: Parsed BeautifulSoup document.
            review_container_selector: CSS selector for each review block.
            author_selector: CSS selector (relative) for the author name.
            rating_selector: CSS selector (relative) for the star rating.
            text_selector: CSS selector (relative) for the review body text.
            max_reviews: Maximum number of reviews to extract.

        Returns:
            A list of ScrapedReview objects (may be empty if none found).
        """
        reviews: list[ScrapedReview] = []
        containers = soup.select(review_container_selector)[:max_reviews]

        for container in containers:
            author_el = container.select_one(author_selector)
            rating_el = container.select_one(rating_selector)
            text_el = container.select_one(text_selector)

            if not text_el:
                continue

            rating_value: float | None = None
            if rating_el:
                rating_value = self._parse_rating_text(rating_el.get_text(strip=True))

            reviews.append(
                ScrapedReview(
                    author=author_el.get_text(strip=True) if author_el else None,
                    rating=rating_value,
                    text=text_el.get_text(strip=True),
                )
            )

        logger.debug(f"Extracted {len(reviews)} reviews from page.")
        return reviews

    @staticmethod
    def _parse_rating_text(rating_text: str) -> float | None:
        """
        Attempt to parse a numeric rating out of free-form text (e.g. "4.5 out of 5").

        Args:
            rating_text: Raw text potentially containing a numeric rating.

        Returns:
            The parsed float rating, or None if no number could be extracted.
        """
        import re

        match = re.search(r"(\d+(\.\d+)?)", rating_text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        return None


_service_instance: ScraperService | None = None


def get_scraper_service() -> ScraperService:
    """Return a lazily-initialized singleton ScraperService instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = ScraperService()
    return _service_instance
