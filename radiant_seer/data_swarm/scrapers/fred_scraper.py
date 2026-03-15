"""FRED (Federal Reserve Economic Data) scraper for macro tensors.

Free API key from https://fred.stlouisfed.org/docs/api/api_key.html
Provides: inflation, rates, unemployment, GDP, and other macro indicators.
"""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
from loguru import logger

from radiant_seer.data_swarm.scrapers.base_scraper import BaseScraper, ScraperResult

# 12 macro series that map to our macro_feature_count=12
# Order matters — must be consistent with training
FRED_SERIES = {
    # Rates cluster (0-2)
    "DFF": "Federal Funds Rate",
    "DGS10": "10-Year Treasury Yield",
    "DGS2": "2-Year Treasury Yield",
    # Inflation cluster (3-5)
    "CPIAUCSL": "CPI (All Urban Consumers)",
    "CPILFESL": "Core CPI (Less Food & Energy)",
    "T5YIE": "5-Year Breakeven Inflation",
    # Employment cluster (6-8)
    "UNRATE": "Unemployment Rate",
    "PAYEMS": "Nonfarm Payrolls",
    "ICSA": "Initial Jobless Claims",
    # Growth/Activity (9-11)
    "GDP": "Gross Domestic Product",
    "INDPRO": "Industrial Production Index",
    "UMCSENT": "Consumer Sentiment (UMich)",
}


class FredScraper(BaseScraper):
    """Fetch macro data from FRED API."""

    def __init__(self, api_key: str | None = None):
        super().__init__(name="fred")
        self.api_key = api_key or os.environ.get("FRED_API_KEY", "")
        self.series_ids = list(FRED_SERIES.keys())

    def fetch(self) -> ScraperResult:
        """Fetch latest values for all FRED series."""
        if not self.api_key:
            return ScraperResult(
                source=self.name,
                timestamp=datetime.now(),
                raw_data={},
                success=False,
                error=(
                    "FRED_API_KEY not set. Get a free key at "
                    "https://fred.stlouisfed.org/docs/api/api_key.html "
                    "and set FRED_API_KEY or SEER_FRED_API_KEY env var."
                ),
            )

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

        values = {}
        errors = []

        for series_id in self.series_ids:
            try:
                url = (
                    f"https://api.stlouisfed.org/fred/series/observations"
                    f"?series_id={series_id}"
                    f"&api_key={self.api_key}"
                    f"&file_type=json"
                    f"&sort_order=desc"
                    f"&limit=1"
                )
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                data = resp.json()

                observations = data.get("observations", [])
                if observations:
                    val = observations[0].get("value", ".")
                    if val != ".":
                        values[series_id] = float(val)
                        continue
                errors.append(f"{series_id}: no data")
            except Exception as e:
                errors.append(f"{series_id}: {e}")

        logger.info(
            f"FRED: fetched {len(values)}/{len(self.series_ids)} series"
        )

        return ScraperResult(
            source=self.name,
            timestamp=datetime.now(),
            raw_data={"values": values, "errors": errors},
            success=len(values) > 0,
            error="; ".join(errors) if errors and not values else None,
            metadata={"n_series": len(values)},
        )

    def transform(self, result: ScraperResult) -> dict:
        """Convert to ordered macro vector matching training feature order.

        Returns:
            Dict with 'macro_values': np.ndarray of shape (12,).
        """
        values = result.raw_data.get("values", {})
        macro = np.zeros(len(self.series_ids), dtype=np.float64)

        for i, series_id in enumerate(self.series_ids):
            macro[i] = values.get(series_id, 0.0)

        return {"macro_values": macro}
