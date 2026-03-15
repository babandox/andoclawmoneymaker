"""Tests for data scrapers and ingestion pipeline."""

import numpy as np

from radiant_seer.data_swarm.news_embedder import NewsEmbedder
from radiant_seer.data_swarm.normalization import StateNormalizer
from radiant_seer.data_swarm.scrapers.base_scraper import ScraperResult
from radiant_seer.data_swarm.scrapers.fred_scraper import FRED_SERIES, FredScraper
from radiant_seer.data_swarm.scrapers.polymarket_scraper import PolymarketScraper
from radiant_seer.data_swarm.scrapers.reddit_scraper import RedditScraper
from radiant_seer.data_swarm.scrapers.rss_scraper import RssScraper


class TestNewsEmbedder:
    def test_fallback_embed(self):
        """Hash fallback produces correct shape embeddings."""
        embedder = NewsEmbedder(dim=384)
        embedder._use_fallback = True
        embedder._model = None

        result = embedder.embed(["Test headline one", "Another headline"])
        assert result.shape == (2, 384)
        # Should be L2-normalized
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_fallback_aggregate(self):
        embedder = NewsEmbedder(dim=384)
        embedder._use_fallback = True
        embedder._model = None

        result = embedder.embed_aggregate(["Headline 1", "Headline 2", "Headline 3"])
        assert result.shape == (384,)
        assert np.linalg.norm(result) > 0

    def test_empty_headlines(self):
        embedder = NewsEmbedder(dim=384)
        embedder._use_fallback = True
        embedder._model = None

        result = embedder.embed([])
        assert result.shape == (0, 384)

        agg = embedder.embed_aggregate([])
        assert agg.shape == (384,)

    def test_deterministic(self):
        embedder = NewsEmbedder(dim=384)
        embedder._use_fallback = True
        embedder._model = None

        r1 = embedder.embed(["Same headline"])
        r2 = embedder.embed(["Same headline"])
        np.testing.assert_array_equal(r1, r2)


class TestStateNormalizer:
    def test_build_state(self):
        norm = StateNormalizer(macro_dim=12, news_dim=384)
        state = norm.build_state(
            news_embedding=np.random.randn(384),
            macro_values=np.random.randn(12),
            sentiment=0.5,
        )
        assert state["news"].shape == (384,)
        assert state["macro"].shape == (12,)
        assert state["sentiment"].shape == (1,)

    def test_fit_and_normalize_macro(self):
        norm = StateNormalizer(macro_dim=12)
        data = np.random.randn(100, 12)
        norm.fit_macro(data)

        result = norm.normalize_macro(data[0])
        # Should be roughly z-scored
        assert result.shape == (12,)

    def test_update_macro(self):
        norm = StateNormalizer(macro_dim=12)
        for _ in range(10):
            norm.update_macro(np.random.randn(12))
        assert norm._macro_count == 10
        assert norm._macro_mean is not None


class TestScraperInterfaces:
    """Test that scrapers handle missing dependencies gracefully."""

    def test_rss_scraper_init(self):
        scraper = RssScraper()
        assert scraper.name == "rss"
        assert len(scraper.feeds) > 0

    def test_fred_scraper_no_key(self):
        scraper = FredScraper(api_key="")
        result = scraper.fetch()
        assert not result.success
        assert "FRED_API_KEY" in result.error

    def test_fred_series_count(self):
        assert len(FRED_SERIES) == 12

    def test_polymarket_scraper_init(self):
        scraper = PolymarketScraper()
        assert scraper.name == "polymarket"

    def test_reddit_scraper_init(self):
        scraper = RedditScraper()
        assert scraper.name == "reddit"
        assert len(scraper.subreddits) > 0

    def test_reddit_transform_empty(self):
        scraper = RedditScraper()
        result = ScraperResult(
            source="reddit",
            timestamp=__import__("datetime").datetime.now(),
            raw_data={"posts": [], "errors": []},
            success=True,
        )
        transformed = scraper.transform(result)
        assert transformed["sentiment"] == 0.0
        assert transformed["headlines"] == []

    def test_reddit_transform_with_posts(self):
        scraper = RedditScraper()
        result = ScraperResult(
            source="reddit",
            timestamp=__import__("datetime").datetime.now(),
            raw_data={
                "posts": [
                    {
                        "title": "Iran threatens to close Strait of Hormuz",
                        "score": 100,
                        "upvote_ratio": 0.9,
                        "num_comments": 50,
                    },
                    {
                        "title": "Oil prices crash on demand fears",
                        "score": 200,
                        "upvote_ratio": 0.3,
                        "num_comments": 100,
                    },
                ],
                "errors": [],
            },
            success=True,
        )
        transformed = scraper.transform(result)
        assert -1.0 <= transformed["sentiment"] <= 1.0
        assert len(transformed["headlines"]) == 2

    def test_rss_transform(self):
        scraper = RssScraper()
        result = ScraperResult(
            source="rss",
            timestamp=__import__("datetime").datetime.now(),
            raw_data={
                "entries": [
                    {"title": "Iran sanctions tightened by US"},
                    {"title": "OPEC cuts crude oil production"},
                    {"title": ""},
                ],
                "errors": [],
            },
            success=True,
        )
        transformed = scraper.transform(result)
        assert len(transformed["headlines"]) == 2  # Empty title filtered

    def test_fred_transform(self):
        scraper = FredScraper(api_key="fake")
        result = ScraperResult(
            source="fred",
            timestamp=__import__("datetime").datetime.now(),
            raw_data={
                "values": {"DFF": 5.33, "DGS10": 4.25, "UNRATE": 3.8},
                "errors": [],
            },
            success=True,
        )
        transformed = scraper.transform(result)
        macro = transformed["macro_values"]
        assert macro.shape == (12,)
        assert macro[0] == 5.33  # DFF is first series
