"""Data ingestion pipeline: scrapers → normalizer → encoder-ready state.

Orchestrates all data sources into a single state tensor dict
that the encoder can consume directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from loguru import logger
from torch import Tensor

from radiant_seer.data_swarm.news_embedder import NewsEmbedder
from radiant_seer.data_swarm.normalization import StateNormalizer
from radiant_seer.data_swarm.scrapers.base_scraper import ScraperResult
from radiant_seer.data_swarm.scrapers.fred_scraper import FredScraper
from radiant_seer.data_swarm.scrapers.polymarket_scraper import PolymarketScraper
from radiant_seer.data_swarm.scrapers.reddit_scraper import RedditScraper
from radiant_seer.data_swarm.scrapers.rss_scraper import RssScraper


@dataclass
class IngestResult:
    state: dict[str, Tensor]
    contracts: dict[str, float]
    questions: dict[str, str]
    timestamp: datetime = field(default_factory=datetime.now)
    headlines_count: int = 0
    macro_available: bool = False
    errors: list[str] = field(default_factory=list)
    headline_embeddings: np.ndarray | None = None  # (N, 384)
    headline_timestamps: list[float] = field(default_factory=list)
    headline_texts: list[str] = field(default_factory=list)
    liquidity: dict[str, float] = field(default_factory=dict)
    volume: dict[str, float] = field(default_factory=dict)


class DataIngestor:
    """Orchestrate all data sources into encoder-ready state."""

    def __init__(
        self,
        normalizer: StateNormalizer,
        fred_api_key: str | None = None,
        contract_ids: list[str] | None = None,
        news_dim: int = 384,
    ):
        self.normalizer = normalizer
        self.news_dim = news_dim

        # Initialize scrapers
        self.rss = RssScraper()
        self.fred = FredScraper(api_key=fred_api_key)
        self.polymarket = PolymarketScraper(contract_ids=contract_ids or [])
        self.reddit = RedditScraper()

        # News embedder
        self.embedder = NewsEmbedder(dim=news_dim)

    def ingest(self) -> IngestResult:
        """Run all scrapers and build a complete state.

        Returns:
            IngestResult with state tensors, contract prices, and metadata.
        """
        errors = []
        headlines = []
        macro_values = np.zeros(self.normalizer.macro_dim, dtype=np.float64)
        macro_available = False
        sentiment = 0.0
        contracts: dict[str, float] = {}
        questions: dict[str, str] = {}

        # 1. News headlines (RSS + Reddit)
        rss_data = self._safe_fetch(self.rss, errors)
        if rss_data:
            headlines.extend(rss_data.get("headlines", []))

        reddit_data = self._safe_fetch(self.reddit, errors)
        if reddit_data:
            headlines.extend(reddit_data.get("headlines", []))
            sentiment = reddit_data.get("sentiment", 0.0)

        # 2. Macro data (FRED)
        fred_data = self._safe_fetch(self.fred, errors)
        if fred_data:
            mv = fred_data.get("macro_values")
            if mv is not None:
                macro_values = mv
                macro_available = True
                self.normalizer.update_macro(macro_values)

        # 3. Market prices (Polymarket)
        liquidity: dict[str, float] = {}
        volume: dict[str, float] = {}
        poly_data = self._safe_fetch(self.polymarket, errors)
        if poly_data:
            contracts = poly_data.get("prices", {})
            questions = poly_data.get("questions", {})
            liquidity = poly_data.get("liquidity", {})
            volume = poly_data.get("volume", {})

        # Build state tensor
        if headlines:
            headline_embs, headline_ts = self.embedder.embed_with_timestamps(headlines)
            news_embedding = headline_embs.mean(axis=0)
            norm = np.linalg.norm(news_embedding)
            if norm > 0:
                news_embedding /= norm
        else:
            headline_embs = np.zeros((0, self.news_dim), dtype=np.float32)
            headline_ts = []
            news_embedding = np.zeros(self.news_dim, dtype=np.float32)
            errors.append("No headlines available — using zero embedding")

        state = self.normalizer.build_state(
            news_embedding=news_embedding,
            macro_values=macro_values,
            sentiment=sentiment,
        )

        if errors:
            for err in errors:
                logger.warning(f"Ingest: {err}")

        logger.info(
            f"Ingest complete: {len(headlines)} headlines, "
            f"macro={'yes' if macro_available else 'no'}, "
            f"{len(contracts)} contracts"
        )

        return IngestResult(
            state=state,
            contracts=contracts,
            questions=questions,
            headlines_count=len(headlines),
            macro_available=macro_available,
            errors=errors,
            headline_embeddings=headline_embs,
            headline_timestamps=headline_ts,
            headline_texts=headlines,
            liquidity=liquidity,
            volume=volume,
        )

    def _safe_fetch(
        self, scraper, errors: list[str]
    ) -> dict | None:
        """Fetch and transform with error handling."""
        try:
            result: ScraperResult = scraper.fetch()
            if not result.success:
                errors.append(f"{scraper.name}: {result.error}")
                return None
            return scraper.transform(result)
        except Exception as e:
            errors.append(f"{scraper.name}: {e}")
            return None
