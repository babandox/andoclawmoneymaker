"""RSS feed scraper for news headlines.

Focused on Iran, oil, energy, and Middle East geopolitics.
No API keys needed.
"""

from __future__ import annotations

from datetime import datetime

from loguru import logger

from radiant_seer.data_swarm.scrapers.base_scraper import BaseScraper, ScraperResult
from radiant_seer.data_swarm.topic_filter import filter_headline

# ── Feeds: Iran, oil, energy, Middle East ────────────────────────────
DEFAULT_FEEDS = [
    # Wire services via Google News (Reuters/AP retired their RSS feeds)
    ("Reuters via GN", "https://news.google.com/rss/search?q=site:reuters.com+iran+OR+oil+OR+gold&hl=en-US&gl=US&ceid=US:en"),
    ("AP via GN", "https://news.google.com/rss/search?q=site:apnews.com+iran+OR+oil+OR+gold+OR+war&hl=en-US&gl=US&ceid=US:en"),
    ("Bloomberg via GN", "https://news.google.com/rss/search?q=site:bloomberg.com+commodities+gold+OR+silver+OR+oil&hl=en-US&gl=US&ceid=US:en"),
    # BBC
    ("BBC World", "https://feeds.bbci.co.uk/news/world/rss.xml"),
    ("BBC Business", "https://feeds.bbci.co.uk/news/business/rss.xml"),
    ("BBC Middle East", "https://feeds.bbci.co.uk/news/world/middle_east/rss.xml"),
    # Middle East / Iran focused
    ("Al Jazeera", "https://www.aljazeera.com/xml/rss/all.xml"),
    ("Iran International", "https://www.iranintl.com/en/feed"),
    ("Times of Israel", "https://www.timesofisrael.com/feed/"),
    ("Jerusalem Post", "https://www.jpost.com/rss/rssfeedsfrontpage.aspx"),
    ("Middle East Eye", "https://www.middleeasteye.net/rss"),
    # Energy / oil
    ("CNBC Economy", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258"),
    ("CNBC World", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100727362"),
    ("MarketWatch", "https://feeds.marketwatch.com/marketwatch/topstories/"),
    ("OilPrice.com", "https://oilprice.com/rss/main"),
    # Gold & silver / precious metals
    ("Kitco via GN", "https://news.google.com/rss/search?q=site:kitco.com+gold+OR+silver&hl=en-US&gl=US&ceid=US:en"),
    ("Seeking Alpha Gold", "https://seekingalpha.com/tag/gold.xml"),
    ("Silver Doctors", "https://www.silverdoctors.com/feed/"),
    ("Gold Telegraph", "https://goldtelegraph.com/feed/"),
    # Targeted topic feeds via Google News
    ("GN Gold/Silver", "https://news.google.com/rss/search?q=gold+price+OR+silver+price+OR+precious+metals&hl=en-US&gl=US&ceid=US:en"),
    ("GN Crude Oil", "https://news.google.com/rss/search?q=crude+oil+price+OR+brent+OR+WTI+OR+OPEC&hl=en-US&gl=US&ceid=US:en"),
    ("GN Iran", "https://news.google.com/rss/search?q=iran+war+OR+iran+nuclear+OR+strait+hormuz&hl=en-US&gl=US&ceid=US:en"),
    # Defense / security
    ("War on the Rocks", "https://warontherocks.com/feed/"),
    # International
    ("Guardian World", "https://www.theguardian.com/world/rss"),
    ("NPR World", "https://feeds.npr.org/1004/rss.xml"),
    # Politics / policy
    ("The Hill", "https://thehill.com/feed/"),
    ("Politico", "https://rss.politico.com/politics-news.xml"),
    ("Investing.com", "https://www.investing.com/rss/news_14.rss"),
]


class RssScraper(BaseScraper):
    """Scrape headlines from RSS feeds, filtered for Iran & oil topics."""

    def __init__(
        self,
        feeds: list[tuple[str, str]] | None = None,
        apply_filter: bool = True,
    ):
        super().__init__(name="rss")
        self.feeds = feeds or DEFAULT_FEEDS
        self.apply_filter = apply_filter

    def fetch(self) -> ScraperResult:
        """Fetch latest headlines from all configured RSS feeds."""
        try:
            import feedparser
        except ImportError:
            return ScraperResult(
                source=self.name,
                timestamp=datetime.now(),
                raw_data={},
                success=False,
                error="feedparser not installed. Run: pip install feedparser",
            )

        import urllib.request

        _UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        all_entries = []
        errors = []

        for feed_name, feed_url in self.feeds:
            try:
                # Some feeds (Google News, Seeking Alpha) require a
                # browser user-agent or they return 403.
                req = urllib.request.Request(feed_url, headers={"User-Agent": _UA})
                resp = urllib.request.urlopen(req, timeout=15)
                feed = feedparser.parse(resp.read())
                if feed.bozo and not feed.entries:
                    errors.append(f"{feed_name}: parse error")
                    continue

                for entry in feed.entries[:50]:
                    all_entries.append({
                        "source": feed_name,
                        "title": entry.get("title", ""),
                        "summary": entry.get("summary", ""),
                        "published": entry.get("published", ""),
                        "link": entry.get("link", ""),
                    })
                logger.debug(
                    f"RSS {feed_name}: {len(feed.entries)} entries"
                )
            except Exception as e:
                errors.append(f"{feed_name}: {e}")

        return ScraperResult(
            source=self.name,
            timestamp=datetime.now(),
            raw_data={"entries": all_entries, "errors": errors},
            success=len(all_entries) > 0,
            error="; ".join(errors) if errors and not all_entries else None,
            metadata={"n_entries": len(all_entries), "n_feeds": len(self.feeds)},
        )

    def transform(self, result: ScraperResult) -> dict:
        """Extract headline strings, filtered for Iran & oil.

        Returns:
            Dict with 'headlines': list of headline strings.
        """
        entries = result.raw_data.get("entries", [])
        headlines = []
        for entry in entries:
            title = entry.get("title", "").strip()
            if not title:
                continue
            if self.apply_filter and not filter_headline(title):
                continue
            headlines.append(title)

        return {"headlines": headlines}
