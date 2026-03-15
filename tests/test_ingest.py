"""Tests for the data ingestion pipeline."""

from unittest.mock import patch

import numpy as np
import pytest

from radiant_seer.data_swarm.ingest import DataIngestor, IngestResult
from radiant_seer.data_swarm.normalization import StateNormalizer
from radiant_seer.data_swarm.scrapers.base_scraper import ScraperResult


@pytest.fixture
def normalizer():
    return StateNormalizer(macro_dim=12, news_dim=384)


@pytest.fixture
def ingestor(normalizer):
    ing = DataIngestor(
        normalizer=normalizer,
        fred_api_key="",
        news_dim=384,
    )
    # Force hash fallback for embedder
    ing.embedder._use_fallback = True
    ing.embedder._model = None
    return ing


class TestDataIngestor:
    def test_ingest_all_scrapers_fail_gracefully(self, ingestor):
        """Pipeline should return valid tensors even if all scrapers fail."""
        # All scrapers will fail (no network, no API keys)
        # but the pipeline should still return a valid state
        with patch.object(ingestor.rss, "fetch") as rss_mock, \
             patch.object(ingestor.fred, "fetch") as fred_mock, \
             patch.object(ingestor.polymarket, "fetch") as poly_mock, \
             patch.object(ingestor.reddit, "fetch") as reddit_mock:

            now = __import__("datetime").datetime.now()
            fail = ScraperResult(
                source="test", timestamp=now, raw_data={},
                success=False, error="test error",
            )
            rss_mock.return_value = fail
            fred_mock.return_value = fail
            poly_mock.return_value = fail
            reddit_mock.return_value = fail

            result = ingestor.ingest()

            assert result.state["news"].shape == (384,)
            assert result.state["macro"].shape == (12,)
            assert result.state["sentiment"].shape == (1,)
            assert len(result.errors) > 0
            assert result.contracts == {}

    def test_ingest_with_mock_data(self, ingestor):
        """Pipeline should produce valid state from mock scraper data."""
        now = __import__("datetime").datetime.now()

        with patch.object(ingestor.rss, "fetch") as rss_mock, \
             patch.object(ingestor.rss, "transform") as rss_trans, \
             patch.object(ingestor.fred, "fetch") as fred_mock, \
             patch.object(ingestor.fred, "transform") as fred_trans, \
             patch.object(ingestor.polymarket, "fetch") as poly_mock, \
             patch.object(ingestor.polymarket, "transform") as poly_trans, \
             patch.object(ingestor.reddit, "fetch") as reddit_mock, \
             patch.object(ingestor.reddit, "transform") as reddit_trans:

            ok = ScraperResult(
                source="test", timestamp=now, raw_data={"ok": True},
                success=True,
            )
            rss_mock.return_value = ok
            fred_mock.return_value = ok
            poly_mock.return_value = ok
            reddit_mock.return_value = ok

            rss_trans.return_value = {
                "headlines": ["Market up", "Fed holds", "GDP grows"]
            }
            fred_trans.return_value = {
                "macro_values": np.array([5.0, 4.2, 3.8, 2.1, 1.9, 2.5,
                                          3.7, 150.0, 210.0, 25000.0, 103.5, 67.2])
            }
            poly_trans.return_value = {
                "prices": {"0xabc": 0.65, "0xdef": 0.40},
                "questions": {"0xabc": "Will X happen?", "0xdef": "Will Y happen?"},
            }
            reddit_trans.return_value = {
                "sentiment": 0.3,
                "headlines": ["Reddit says bullish"],
            }

            result = ingestor.ingest()

            assert result.state["news"].shape == (384,)
            assert result.state["macro"].shape == (12,)
            assert result.state["sentiment"].shape == (1,)
            assert result.headlines_count == 4  # 3 RSS + 1 Reddit
            assert result.macro_available
            assert len(result.contracts) == 2
            assert result.contracts["0xabc"] == 0.65

    def test_ingest_result_dataclass(self):
        result = IngestResult(
            state={"news": None, "macro": None, "sentiment": None},
            contracts={"abc": 0.5},
            questions={"abc": "Test?"},
        )
        assert result.headlines_count == 0
        assert len(result.errors) == 0
