"""Reddit scraper for rough social sentiment.

Uses Reddit's public JSON API (no auth needed).
Focused on geopolitics, economics, and politics subreddits.
"""

from __future__ import annotations

from datetime import datetime

from loguru import logger

from radiant_seer.data_swarm.scrapers.base_scraper import BaseScraper, ScraperResult
from radiant_seer.data_swarm.topic_filter import filter_headline

# Subreddits covering Iran, oil, energy, Middle East
DEFAULT_SUBREDDITS = [
    "worldnews",
    "geopolitics",
    "MiddleEastNews",
    "iran",
    "energy",
    "oil",
    "economics",
    "commodities",
    "Gold",
    "Silverbugs",
    "wallstreetsilver",
]

USER_AGENT = "RadiantSeer/0.1 (research bot)"


class RedditScraper(BaseScraper):
    """Scrape top posts from Reddit for sentiment analysis."""

    def __init__(
        self,
        subreddits: list[str] | None = None,
        apply_filter: bool = True,
    ):
        super().__init__(name="reddit")
        self.subreddits = subreddits or DEFAULT_SUBREDDITS
        self.apply_filter = apply_filter

    def fetch(self) -> ScraperResult:
        """Fetch top posts from configured subreddits."""
        try:
            import requests
        except ImportError:
            return ScraperResult(
                source=self.name,
                timestamp=datetime.now(),
                raw_data={},
                success=False,
                error="requests not installed. Run: pip install requests",
            )

        posts = []
        errors = []

        for sub in self.subreddits:
            try:
                url = f"https://www.reddit.com/r/{sub}/hot.json"
                resp = requests.get(
                    url,
                    headers={"User-Agent": USER_AGENT},
                    params={"limit": 15},
                    timeout=10,
                )
                resp.raise_for_status()
                data = resp.json()

                children = data.get("data", {}).get("children", [])
                for child in children:
                    post = child.get("data", {})
                    posts.append({
                        "subreddit": sub,
                        "title": post.get("title", ""),
                        "score": post.get("score", 0),
                        "upvote_ratio": post.get("upvote_ratio", 0.5),
                        "num_comments": post.get("num_comments", 0),
                        "created_utc": post.get("created_utc", 0),
                    })

                logger.debug(f"Reddit r/{sub}: {len(children)} posts")
            except Exception as e:
                errors.append(f"r/{sub}: {e}")

        return ScraperResult(
            source=self.name,
            timestamp=datetime.now(),
            raw_data={"posts": posts, "errors": errors},
            success=len(posts) > 0,
            error="; ".join(errors) if errors and not posts else None,
            metadata={"n_posts": len(posts), "n_subs": len(self.subreddits)},
        )

    def transform(self, result: ScraperResult) -> dict:
        """Extract sentiment signal from Reddit posts, filtered for signal topics.

        Returns:
            Dict with:
              'sentiment': float in [-1, 1]
              'headlines': list of post titles (for embedding)
        """
        posts = result.raw_data.get("posts", [])
        if not posts:
            return {"sentiment": 0.0, "headlines": []}

        total_weight = 0.0
        weighted_sentiment = 0.0
        headlines = []

        for post in posts:
            title = post.get("title", "").strip()
            if not title:
                continue
            if self.apply_filter and not filter_headline(title):
                continue

            headlines.append(title)

            score = max(1, post.get("score", 1))
            ratio = post.get("upvote_ratio", 0.5)
            weight = min(score, 10000)

            sent = (ratio - 0.5) * 2.0
            weighted_sentiment += weight * sent
            total_weight += weight

        sentiment = weighted_sentiment / total_weight if total_weight > 0 else 0.0
        sentiment = max(-1.0, min(1.0, sentiment))

        return {"sentiment": sentiment, "headlines": headlines}
