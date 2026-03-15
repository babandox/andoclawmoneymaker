"""Tests for data collector and snapshot loading."""

import json
from unittest.mock import patch

import pytest

from radiant_seer.data_swarm.collector import DataCollector, load_snapshots
from radiant_seer.data_swarm.scrapers.base_scraper import ScraperResult


@pytest.fixture
def tmp_snapshot_dir(tmp_path):
    return tmp_path / "snapshots"


@pytest.fixture
def collector(tmp_snapshot_dir):
    c = DataCollector(output_dir=tmp_snapshot_dir, fred_api_key="")
    # Force hash fallback
    c.embedder._use_fallback = True
    c.embedder._model = None
    return c


class TestDataCollector:
    def test_collect_snapshot_with_mocked_scrapers(
        self, collector, tmp_snapshot_dir
    ):
        """Snapshot saves even when scrapers partially fail."""
        now = __import__("datetime").datetime.now()
        ok = ScraperResult(
            source="rss", timestamp=now,
            raw_data={
                "entries": [{"title": "Iran oil exports hit by new sanctions"}],
                "errors": [],
            },
            success=True,
        )
        fail = ScraperResult(
            source="fail", timestamp=now, raw_data={},
            success=False, error="no key",
        )

        with patch.object(collector.rss, "fetch", return_value=ok), \
             patch.object(collector.fred, "fetch", return_value=fail), \
             patch.object(collector.polymarket, "fetch", return_value=fail), \
             patch.object(collector.reddit, "fetch", return_value=fail):

            path = collector.collect_snapshot()

        assert path is not None
        assert path.exists()

        with open(path) as f:
            snap = json.load(f)

        assert snap["n_headlines"] == 1
        assert len(snap["news_embedding"]) == 384
        assert len(snap["macro_values"]) == 12
        assert "timestamp" in snap

    def test_collect_all_fail(self, collector, tmp_snapshot_dir):
        """Even total failure produces a valid snapshot."""
        now = __import__("datetime").datetime.now()
        fail = ScraperResult(
            source="fail", timestamp=now, raw_data={},
            success=False, error="fail",
        )

        with patch.object(collector.rss, "fetch", return_value=fail), \
             patch.object(collector.fred, "fetch", return_value=fail), \
             patch.object(collector.polymarket, "fetch", return_value=fail), \
             patch.object(collector.reddit, "fetch", return_value=fail):

            path = collector.collect_snapshot()

        assert path is not None
        with open(path) as f:
            snap = json.load(f)
        assert snap["n_headlines"] == 0


class TestLoadSnapshots:
    def test_load_empty_dir(self, tmp_path):
        assert load_snapshots(tmp_path) == []

    def test_load_saved_snapshots(self, tmp_path):
        for i in range(3):
            snap = {
                "timestamp": f"2026-03-{10+i}T12:00:00",
                "news_embedding": [0.0] * 384,
                "macro_values": [0.0] * 12,
                "sentiment": 0.0,
                "n_headlines": 5,
            }
            path = tmp_path / f"snapshot_202603{10+i}_120000.json"
            with open(path, "w") as f:
                json.dump(snap, f)

        loaded = load_snapshots(tmp_path)
        assert len(loaded) == 3
        # Should be sorted by filename (timestamp)
        assert "2026-03-10" in loaded[0]["timestamp"]
