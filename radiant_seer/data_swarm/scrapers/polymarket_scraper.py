"""Polymarket scraper for market prices and contract data.

Uses the events endpoint with tag_slug filtering to find Iran & oil markets.
No API key needed for reading public market data.
"""

from __future__ import annotations

import json
from datetime import datetime

from loguru import logger

from radiant_seer.data_swarm.scrapers.base_scraper import BaseScraper, ScraperResult
from radiant_seer.data_swarm.topic_filter import filter_contract

GAMMA_API = "https://gamma-api.polymarket.com"

# Tag slugs that contain Iran/oil/energy/geopolitics markets
TARGET_TAG_SLUGS = [
    "oil",
    "iran",
    "middle-east",
    "geopolitics",
    "energy",
    "commodities",
    "crude",
    "petroleum",
    "gold",
    "silver",
]


class PolymarketScraper(BaseScraper):
    """Fetch market prices from Polymarket, focused on Iran & oil.

    Args:
        contract_ids: Specific contract IDs to always include.
        apply_filter: Whether to apply topic filtering in transform().
        tag_slugs: Tag slugs to query (default: Iran/oil/geopolitics).
        fetch_all: If True, fetch ALL active markets (ignores tag_slugs).
    """

    def __init__(
        self,
        contract_ids: list[str] | None = None,
        apply_filter: bool = True,
        tag_slugs: list[str] | None = None,
        fetch_all: bool = False,
    ):
        super().__init__(name="polymarket")
        self.contract_ids = contract_ids or []
        self.apply_filter = apply_filter
        self.fetch_all = fetch_all
        self.tag_slugs = tag_slugs or TARGET_TAG_SLUGS

    def _fetch_events_by_tag(self, tag_slug: str, requests_mod) -> list[dict]:
        """Fetch active events for a tag slug, paginated."""
        events = []
        offset = 0
        for _ in range(5):  # Max 5 pages
            resp = requests_mod.get(
                f"{GAMMA_API}/events",
                params={
                    "active": "true",
                    "closed": "false",
                    "limit": 50,
                    "offset": offset,
                    "tag_slug": tag_slug,
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            if not data:
                break
            events.extend(data)
            offset += 50
            if len(data) < 50:
                break
        return events

    def _fetch_all_events(self, requests_mod) -> list[dict]:
        """Fetch all active events without tag filtering, paginated."""
        events = []
        offset = 0
        for _ in range(40):  # Up to 2000 events
            resp = requests_mod.get(
                f"{GAMMA_API}/events",
                params={
                    "active": "true",
                    "closed": "false",
                    "limit": 50,
                    "offset": offset,
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            if not data:
                break
            events.extend(data)
            offset += 50
            if len(data) < 50:
                break
        return events

    def fetch(self) -> ScraperResult:
        """Fetch markets from events endpoint.

        If fetch_all=True, gets all active markets.
        Otherwise, queries by tag slugs.
        """
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

        seen_cids: set[str] = set()
        markets = []
        events_found = 0
        errors = []

        if self.fetch_all:
            # Fetch ALL active events without tag filtering
            try:
                all_events = self._fetch_all_events(requests)
                events_found = len(all_events)
                for event in all_events:
                    for market in event.get("markets", []):
                        cid = market.get("conditionId", "")
                        if not cid or cid in seen_cids:
                            continue
                        if not market.get("active") or market.get("closed"):
                            continue
                        seen_cids.add(cid)

                        market_info = {
                            "condition_id": cid,
                            "question": market.get("question", ""),
                            "event_title": event.get("title", ""),
                            "outcome_prices": market.get("outcomePrices", ""),
                            "volume": market.get("volume", 0),
                            "liquidity": market.get("liquidity", 0),
                            "end_date": market.get("endDate", ""),
                        }

                        prices_str = market_info["outcome_prices"]
                        if isinstance(prices_str, str) and prices_str:
                            try:
                                prices = json.loads(prices_str)
                                if len(prices) >= 1:
                                    market_info["p_yes"] = float(prices[0])
                                if len(prices) >= 2:
                                    market_info["p_no"] = float(prices[1])
                            except (json.JSONDecodeError, ValueError, IndexError):
                                pass

                        markets.append(market_info)

                logger.info(
                    f"Polymarket (all markets): {len(markets)} live markets "
                    f"from {events_found} events"
                )
            except Exception as e:
                errors.append(f"fetch_all: {e}")
        else:
            # Original tag-based fetching
            for tag_slug in self.tag_slugs:
                try:
                    events = self._fetch_events_by_tag(tag_slug, requests)
                    events_found += len(events)

                    for event in events:
                        for market in event.get("markets", []):
                            cid = market.get("conditionId", "")
                            if not cid or cid in seen_cids:
                                continue
                            if not market.get("active") or market.get("closed"):
                                continue
                            seen_cids.add(cid)

                            market_info = {
                                "condition_id": cid,
                                "question": market.get("question", ""),
                                "event_title": event.get("title", ""),
                                "outcome_prices": market.get("outcomePrices", ""),
                                "volume": market.get("volume", 0),
                                "liquidity": market.get("liquidity", 0),
                                "end_date": market.get("endDate", ""),
                            }

                            prices_str = market_info["outcome_prices"]
                            if isinstance(prices_str, str) and prices_str:
                                try:
                                    prices = json.loads(prices_str)
                                    if len(prices) >= 1:
                                        market_info["p_yes"] = float(prices[0])
                                    if len(prices) >= 2:
                                        market_info["p_no"] = float(prices[1])
                                except (json.JSONDecodeError, ValueError, IndexError):
                                    pass

                            markets.append(market_info)

                    logger.debug(
                        f"Polymarket tag_slug={tag_slug}: "
                        f"{len(events)} events"
                    )
                except Exception as e:
                    errors.append(f"tag_slug {tag_slug}: {e}")

            logger.info(
                f"Polymarket: {len(markets)} live markets "
                f"from {events_found} events across {len(self.tag_slugs)} tags"
            )

        # Fetch specific contracts if configured (always included)
        specific = {}
        for cid in self.contract_ids:
            if cid in seen_cids:
                continue
            try:
                resp = requests.get(
                    f"{GAMMA_API}/markets/{cid}",
                    timeout=10,
                )
                if resp.status_code == 200:
                    m = resp.json()
                    prices_str = m.get("outcomePrices", "")
                    p_yes = None
                    if isinstance(prices_str, str) and prices_str:
                        try:
                            prices = json.loads(prices_str)
                            if prices:
                                p_yes = float(prices[0])
                        except (json.JSONDecodeError, ValueError):
                            pass
                    specific[cid] = {
                        "question": m.get("question", ""),
                        "p_yes": p_yes,
                        "volume": m.get("volume", 0),
                    }
            except Exception as e:
                errors.append(f"contract {cid}: {e}")

        return ScraperResult(
            source=self.name,
            timestamp=datetime.now(),
            raw_data={
                "markets": markets,
                "specific": specific,
                "errors": errors,
            },
            success=len(markets) > 0 or len(specific) > 0,
            error="; ".join(errors) if errors else None,
            metadata={
                "n_markets": len(markets),
                "n_specific": len(specific),
                "n_events": events_found,
            },
        )

    def transform(self, result: ScraperResult) -> dict:
        """Extract contract prices, filtered for Iran & oil.

        Returns:
            Dict with:
              'prices': dict mapping condition_id -> P(YES)
              'questions': dict mapping condition_id -> question text
              'liquidity': dict mapping condition_id -> liquidity in USD
              'volume': dict mapping condition_id -> total volume traded
        """
        prices = {}
        questions = {}
        liquidity = {}
        volume = {}

        now_iso = datetime.now().isoformat()

        for market in result.raw_data.get("markets", []):
            cid = market.get("condition_id", "")
            p_yes = market.get("p_yes")
            question = market.get("question", "")
            if not cid or p_yes is None:
                continue
            if self.apply_filter and not filter_contract(question):
                continue
            # Skip contracts whose end date has passed
            end_date = market.get("end_date", "")
            if end_date and end_date < now_iso:
                continue
            prices[cid] = p_yes
            questions[cid] = question
            liq = market.get("liquidity", 0)
            liquidity[cid] = float(liq) if liq else 0.0
            vol = market.get("volume", 0)
            volume[cid] = float(vol) if vol else 0.0

        for cid, info in result.raw_data.get("specific", {}).items():
            if info.get("p_yes") is not None:
                end_date = info.get("end_date", "")
                if end_date and end_date < now_iso:
                    continue
                prices[cid] = info["p_yes"]
                questions[cid] = info.get("question", "")
                liquidity[cid] = float(info.get("liquidity", 0) or 0)
                volume[cid] = float(info.get("volume", 0) or 0)

        return {
            "prices": prices,
            "questions": questions,
            "liquidity": liquidity,
            "volume": volume,
        }
