"""Trump / Truth Social scraper via Google News RSS.

Truth Social's API is locked down (403 on all endpoints), so we pull
Trump's statements via Google News RSS which aggregates coverage of
his Truth Social posts from major outlets.

This captures the market-moving signal: when Trump posts about Iran,
tariffs, sanctions, or oil, news outlets report it within minutes.
"""

from __future__ import annotations

from datetime import datetime

from loguru import logger

from radiant_seer.data_swarm.scrapers.base_scraper import BaseScraper, ScraperResult

# Google News RSS for Trump statements — covers Truth Social posts
# reported by Reuters, AP, BBC, WashPost, The Hill, etc.
TRUMP_NEWS_FEED = (
    "https://news.google.com/rss/search?"
    "q=Trump+Truth+Social+OR+Trump+statement+OR+Trump+announced"
    "&hl=en-US&gl=US&ceid=US:en"
)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)


class TruthSocialScraper(BaseScraper):
    """Scrape Trump's statements via Google News RSS feed."""

    def __init__(self, feed_url: str = TRUMP_NEWS_FEED):
        super().__init__(name="truthsocial")
        self.feed_url = feed_url

    def fetch(self) -> ScraperResult:
        """Fetch recent Trump/Truth Social coverage from Google News."""
        try:
            import feedparser
        except ImportError:
            return ScraperResult(
                source=self.name,
                timestamp=datetime.now(),
                raw_data={},
                success=False,
                error="feedparser not installed",
            )

        try:
            feed = feedparser.parse(
                self.feed_url,
                agent=USER_AGENT,
            )

            if feed.bozo and not feed.entries:
                return ScraperResult(
                    source=self.name,
                    timestamp=datetime.now(),
                    raw_data={},
                    success=False,
                    error=f"Feed parse error: {feed.bozo_exception}",
                )

            entries = []
            for entry in feed.entries[:40]:
                entries.append({
                    "title": entry.get("title", ""),
                    "source": entry.get("source", {}).get("title", ""),
                    "published": entry.get("published", ""),
                    "link": entry.get("link", ""),
                })

            logger.debug(f"Truth Social (Google News): {len(entries)} entries")

            return ScraperResult(
                source=self.name,
                timestamp=datetime.now(),
                raw_data={"entries": entries},
                success=len(entries) > 0,
                metadata={"n_entries": len(entries)},
            )
        except Exception as e:
            return ScraperResult(
                source=self.name,
                timestamp=datetime.now(),
                raw_data={},
                success=False,
                error=str(e),
            )

    def transform(self, result: ScraperResult) -> dict:
        """Extract headlines from Trump coverage.

        Returns:
            Dict with 'headlines': list of headline strings.
        """
        entries = result.raw_data.get("entries", [])
        headlines = []

        for entry in entries:
            title = entry.get("title", "").strip()
            if not title:
                continue
            # Google News appends " - Source Name", keep it for context
            headlines.append(title)

        return {"headlines": headlines}
