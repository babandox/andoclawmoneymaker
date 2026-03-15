"""Base scraper interface. All data scrapers inherit from this."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ScraperResult:
    source: str
    timestamp: datetime
    raw_data: dict
    success: bool = True
    error: str | None = None
    metadata: dict = field(default_factory=dict)


class BaseScraper(ABC):
    """Abstract base class for all data scrapers."""

    def __init__(self, name: str):
        self.name = name
        self._last_fetch: datetime | None = None

    @abstractmethod
    def fetch(self) -> ScraperResult:
        """Fetch raw data from the source.

        Returns:
            ScraperResult with raw data or error information.
        """
        ...

    @abstractmethod
    def transform(self, result: ScraperResult) -> dict:
        """Transform raw data into normalized format for the pipeline.

        Args:
            result: Raw ScraperResult from fetch().

        Returns:
            Dict with keys matching expected pipeline inputs
            (e.g., 'headlines', 'macro_values', 'sentiment', 'prices').
        """
        ...

    def fetch_and_transform(self) -> dict:
        """Convenience method: fetch then transform."""
        result = self.fetch()
        if not result.success:
            raise RuntimeError(f"Scraper {self.name} failed: {result.error}")
        self._last_fetch = result.timestamp
        return self.transform(result)
