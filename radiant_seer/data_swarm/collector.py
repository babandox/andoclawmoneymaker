"""Data collector: snapshots live scraper output to disk for training.

Saves timestamped snapshots of (news_embedding, macro_values, sentiment,
contract_prices) so we can retrain the encoder/predictor on real data.

Usage:
    python -m radiant_seer.data_swarm.collector              # one snapshot
    python -m radiant_seer.data_swarm.collector --loop 900   # every 15 min
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from loguru import logger

from radiant_seer.data_swarm.news_embedder import NewsEmbedder
from radiant_seer.data_swarm.scrapers.fred_scraper import FredScraper
from radiant_seer.data_swarm.scrapers.polymarket_scraper import PolymarketScraper
from radiant_seer.data_swarm.scrapers.reddit_scraper import RedditScraper
from radiant_seer.data_swarm.scrapers.rss_scraper import RssScraper
from radiant_seer.data_swarm.sentiment import HeadlineSentimentAnalyzer


class DataCollector:
    """Collect and save live data snapshots for offline training."""

    def __init__(
        self,
        output_dir: Path | None = None,
        fred_api_key: str | None = None,
        news_dim: int = 384,
    ):
        self.output_dir = output_dir or (
            Path(__file__).resolve().parent.parent.parent / "data" / "snapshots"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.rss = RssScraper()
        self.fred = FredScraper(
            api_key=fred_api_key or os.environ.get("FRED_API_KEY", "")
        )
        self.polymarket = PolymarketScraper()
        self.reddit = RedditScraper()
        self.embedder = NewsEmbedder(dim=news_dim)
        self.sentiment_analyzer = HeadlineSentimentAnalyzer(self.embedder)

    def collect_snapshot(self) -> Path | None:
        """Collect one data snapshot and save to disk.

        Returns:
            Path to saved snapshot, or None on failure.
        """
        ts = datetime.now()
        ts_str = ts.strftime("%Y%m%d_%H%M%S")
        snapshot: dict = {"timestamp": ts.isoformat()}

        # News headlines
        headlines = []
        try:
            rss_result = self.rss.fetch()
            if rss_result.success:
                rss_data = self.rss.transform(rss_result)
                headlines.extend(rss_data.get("headlines", []))
        except Exception as e:
            logger.warning(f"RSS fetch failed: {e}")

        try:
            reddit_result = self.reddit.fetch()
            if reddit_result.success:
                reddit_data = self.reddit.transform(reddit_result)
                headlines.extend(reddit_data.get("headlines", []))
        except Exception as e:
            logger.warning(f"Reddit fetch failed: {e}")

        snapshot["headlines"] = headlines
        snapshot["n_headlines"] = len(headlines)

        # Content-based sentiment on all headlines
        if headlines:
            snapshot["sentiment"] = self.sentiment_analyzer.score_headlines(
                headlines
            )
        else:
            snapshot["sentiment"] = 0.0

        # Embed headlines
        if headlines:
            news_emb = self.embedder.embed_aggregate(headlines)
        else:
            news_emb = np.zeros(self.embedder.dim, dtype=np.float32)
        snapshot["news_embedding"] = news_emb.tolist()

        # Macro data
        try:
            fred_result = self.fred.fetch()
            if fred_result.success:
                fred_data = self.fred.transform(fred_result)
                snapshot["macro_values"] = fred_data["macro_values"].tolist()
                snapshot["macro_available"] = True
            else:
                snapshot["macro_values"] = [0.0] * 12
                snapshot["macro_available"] = False
                logger.warning(f"FRED: {fred_result.error}")
        except Exception as e:
            snapshot["macro_values"] = [0.0] * 12
            snapshot["macro_available"] = False
            logger.warning(f"FRED fetch failed: {e}")

        # Market prices
        try:
            poly_result = self.polymarket.fetch()
            if poly_result.success:
                poly_data = self.polymarket.transform(poly_result)
                snapshot["contracts"] = poly_data["prices"]
                snapshot["questions"] = poly_data["questions"]
                snapshot["liquidity"] = poly_data.get("liquidity", {})
                snapshot["volume"] = poly_data.get("volume", {})
            else:
                snapshot["contracts"] = {}
                snapshot["questions"] = {}
                snapshot["liquidity"] = {}
                snapshot["volume"] = {}
        except Exception as e:
            snapshot["contracts"] = {}
            snapshot["questions"] = {}
            snapshot["liquidity"] = {}
            snapshot["volume"] = {}
            logger.warning(f"Polymarket fetch failed: {e}")

        # Append new headlines to the deduplicated headline store
        n_new = append_headlines(
            headlines, self.output_dir.parent / "headlines.jsonl"
        )

        # Save slim snapshot (no raw headlines — they're in headlines.jsonl)
        slim_keys = ("headline_texts", "headlines")
        slim = {k: v for k, v in snapshot.items() if k not in slim_keys}
        path = self.output_dir / f"snapshot_{ts_str}.json"
        with open(path, "w") as f:
            json.dump(slim, f, indent=2)

        logger.info(
            f"Snapshot saved: {path.name} | "
            f"{len(headlines)} headlines ({n_new} new), "
            f"macro={'yes' if snapshot.get('macro_available') else 'no'}, "
            f"{len(snapshot.get('contracts', {}))} contracts"
        )
        return path

    def collect_loop(self, interval_seconds: int = 900) -> None:
        """Collect snapshots on a timer."""
        logger.info(
            f"Starting data collection loop — every {interval_seconds}s"
        )
        count = 0
        try:
            while True:
                self.collect_snapshot()
                count += 1
                logger.info(
                    f"Snapshots collected: {count}. "
                    f"Next in {interval_seconds}s."
                )
                time.sleep(interval_seconds)
        except KeyboardInterrupt:
            logger.info(f"Collection stopped. {count} snapshots saved.")


def append_headlines(
    headlines: list[str],
    headline_path: Path | None = None,
) -> int:
    """Append new unique headlines to the deduplicated headline store.

    Returns:
        Number of newly added headlines.
    """
    headline_path = headline_path or (
        Path(__file__).resolve().parent.parent.parent / "data" / "headlines.jsonl"
    )

    # Load existing headlines for dedup
    seen: set[str] = set()
    if headline_path.exists():
        with open(headline_path) as f:
            for line in f:
                rec = json.loads(line)
                seen.add(rec["text"])

    # Append only new ones
    new_count = 0
    ts = datetime.now().isoformat()
    with open(headline_path, "a") as f:
        for h in headlines:
            if h not in seen:
                seen.add(h)
                f.write(json.dumps({"text": h, "first_seen": ts}) + "\n")
                new_count += 1

    return new_count


def load_headlines(
    headline_path: Path | None = None,
) -> list[dict]:
    """Load all headlines from the deduplicated store.

    Returns:
        List of {"text": str, "first_seen": str} dicts.
    """
    headline_path = headline_path or (
        Path(__file__).resolve().parent.parent.parent / "data" / "headlines.jsonl"
    )
    headlines = []
    if headline_path.exists():
        with open(headline_path) as f:
            for line in f:
                headlines.append(json.loads(line))
    return headlines


def load_snapshots(
    snapshot_dir: Path | None = None,
) -> list[dict]:
    """Load all saved snapshots from disk.

    Handles both legacy snapshots (with inline headlines) and slim
    snapshots (headlines stored separately in headlines.jsonl).

    Returns:
        List of snapshot dicts sorted by timestamp.
    """
    snapshot_dir = snapshot_dir or (
        Path(__file__).resolve().parent.parent.parent / "data" / "snapshots"
    )
    snapshots = []
    for path in sorted(snapshot_dir.glob("snapshot_*.json")):
        with open(path) as f:
            snapshots.append(json.load(f))
    return snapshots


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect data snapshots")
    parser.add_argument(
        "--loop", type=int, default=0,
        help="Loop interval in seconds (0 = single snapshot)",
    )
    args = parser.parse_args()

    collector = DataCollector()
    if args.loop > 0:
        collector.collect_loop(interval_seconds=args.loop)
    else:
        collector.collect_snapshot()
