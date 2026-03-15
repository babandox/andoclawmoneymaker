"""Radiant Seer — Live Dashboard.

Rich-based terminal UI that runs the data collector and displays
real-time status: scraper health, headlines, macro data, market prices,
sentiment, collection history, and live price charts for oil and Iran war.

Usage:
    python -m radiant_seer.dashboard
    python -m radiant_seer.dashboard --interval 300
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from rich.align import Align
from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from radiant_seer.configuration.settings import SeerConfig
from radiant_seer.execution.kelly_sizing import KellySizer
from radiant_seer.data_swarm.news_embedder import NewsEmbedder
from radiant_seer.data_swarm.scrapers.fred_scraper import FRED_SERIES, FredScraper
from radiant_seer.data_swarm.scrapers.polymarket_scraper import PolymarketScraper
from radiant_seer.data_swarm.scrapers.reddit_scraper import RedditScraper
from radiant_seer.data_swarm.scrapers.rss_scraper import DEFAULT_FEEDS, RssScraper
from radiant_seer.data_swarm.scrapers.truthsocial_scraper import TruthSocialScraper
from radiant_seer.data_swarm.sentiment import HeadlineSentimentAnalyzer
from radiant_seer.intelligence.contract_decoder import (
    ContractDecoder,
    ContractDecoderV2,
    ContractHistory,
)
from radiant_seer.intelligence.multimodal_encoder import MultimodalEncoder
from radiant_seer.intelligence.relevance import (
    CausalDomainGraph,
    LearnedRelevanceScorer,
    RelevanceRouter,
)
from radiant_seer.learning import OnlineLearner, OnlineLearnerV2
from radiant_seer.planning.reward_module import OutcomeDecoder
from radiant_seer.scanner import EDGE_BUCKETS, Prediction, Scorecard

# ── Colour palette ───────────────────────────────────────────────────
ACCENT = "cyan"
GOOD = "green"
WARN = "yellow"
BAD = "red"
DIM = "bright_black"
TITLE_STYLE = "bold cyan"
OIL_COLOR = "bright_yellow"
WAR_COLOR = "bright_red"

# FRED series display names (short)
FRED_SHORT = {
    "DFF": "Fed Funds",
    "DGS10": "10Y Yield",
    "DGS2": "2Y Yield",
    "CPIAUCSL": "CPI",
    "CPILFESL": "Core CPI",
    "T5YIE": "5Y Breakeven",
    "UNRATE": "Unemployment",
    "PAYEMS": "Payrolls",
    "ICSA": "Jobless Claims",
    "GDP": "GDP",
    "INDPRO": "Industrial Prod",
    "UMCSENT": "Consumer Sent",
}

# ── Chart config ─────────────────────────────────────────────────────
# Braille-based sparkline characters for smooth charts
CHART_BLOCKS = " " + "".join(chr(c) for c in range(0x2581, 0x2589))  # ▁▂▃▄▅▆▇█
MAX_CHART_HISTORY = 60  # Max data points to show

# Patterns to auto-detect key contracts
OIL_PRICE_RE = re.compile(
    r"Crude Oil.*settle.*\$(\d+)", re.IGNORECASE
)
OIL_HIT_RE = re.compile(
    r"Crude Oil.*hit.*\$(1[01]\d|[89]\d).*(?:March|June)", re.IGNORECASE
)
IRAN_WAR_RE = re.compile(
    r"(?:military action.*iran|iran.*conflict|US.*Iran.*ceasefire"
    r"|iran.*leadership|iranian regime.*fall|U\.S\.\s*invade\s*iran"
    r"|strait of hormuz|iran.*strike|strike.*iran)",
    re.IGNORECASE,
)


class PriceHistory:
    """Rolling price history for a tracked contract."""

    def __init__(self, label: str, maxlen: int = MAX_CHART_HISTORY):
        self.label = label
        self.prices: deque[float] = deque(maxlen=maxlen)
        self.timestamps: deque[str] = deque(maxlen=maxlen)

    def add(self, price: float, ts: str | None = None) -> None:
        self.prices.append(price)
        self.timestamps.append(ts or datetime.now().strftime("%H:%M"))

    @property
    def current(self) -> float | None:
        return self.prices[-1] if self.prices else None

    @property
    def change(self) -> float | None:
        if len(self.prices) < 2:
            return None
        return self.prices[-1] - self.prices[0]

    def sparkline(self, width: int = 50) -> str:
        """Render a Unicode block sparkline."""
        if len(self.prices) < 2:
            return ""
        values = list(self.prices)
        # Resample to width if needed
        if len(values) > width:
            step = len(values) / width
            values = [values[int(i * step)] for i in range(width)]
        elif len(values) < width:
            # Pad left with first value
            values = [values[0]] * (width - len(values)) + values

        lo = min(values)
        hi = max(values)
        span = hi - lo if hi > lo else 1e-6

        chars = []
        for v in values:
            idx = int((v - lo) / span * (len(CHART_BLOCKS) - 1))
            idx = max(0, min(len(CHART_BLOCKS) - 1, idx))
            chars.append(CHART_BLOCKS[idx])
        return "".join(chars)


class DashboardState:
    """Mutable state container for the dashboard."""

    def __init__(self):
        self.cycle_count: int = 0
        self.last_collect: datetime | None = None
        self.next_collect: datetime | None = None
        self.collecting: bool = False

        # Scraper status
        self.rss_ok: bool = False
        self.rss_count: int = 0
        self.reddit_ok: bool = False
        self.reddit_count: int = 0
        self.fred_ok: bool = False
        self.fred_count: int = 0
        self.poly_ok: bool = False
        self.poly_count: int = 0
        self.truth_ok: bool = False
        self.truth_count: int = 0

        # Data
        self.headlines: list[str] = []
        self.sentiment: float = 0.0
        self.macro_values: dict[str, float] = {}
        self.contracts: dict[str, float] = {}
        self.questions: dict[str, str] = {}
        self.liquidity: dict[str, float] = {}

        # History
        self.snapshots_on_disk: int = 0
        self.errors: list[str] = []
        self.collect_times: list[float] = []

        # Headline dedup
        self._seen_headlines: set[str] = set()
        self.new_headlines: list[str] = []  # headlines new this cycle
        self.total_unique_headlines: int = 0

        # Charts — keyed by contract question substring
        self.oil_tracks: list[PriceHistory] = []
        self.war_tracks: list[PriceHistory] = []

        # Predictions
        self.p_model: float | None = None
        self.p_model_range: tuple[float, float] | None = None
        self.pending_predictions: dict[str, Prediction] = {}
        self.scorecard: Scorecard = Scorecard()
        self.cycle_pnl: float = 0.0
        self.cycle_scored: int = 0
        self.cycle_correct: int = 0
        self.predictions_active: bool = False

        # Learning
        self.learning_active: bool = False
        self.learning_loss: float | None = None
        self.learning_buffer_size: int = 0
        self.learning_total_steps: int = 0
        self.last_snapshot_data: dict | None = None  # state at prediction time

        # Per-contract history (for V2 decoder)
        self.contract_history: dict[str, ContractHistory] = {}

    def update_charts(self) -> None:
        """Auto-detect and track key oil/Iran contracts."""
        ts = datetime.now().strftime("%H:%M")

        # ── Oil chart: pick the most interesting crude oil contracts ──
        oil_candidates: list[tuple[str, str, float]] = []
        for cid, q in self.questions.items():
            p = self.contracts.get(cid)
            if p is None:
                continue
            if OIL_PRICE_RE.search(q) or OIL_HIT_RE.search(q):
                oil_candidates.append((cid, q, p))

        # Pick up to 5 most informative (prices nearest 50% = most uncertain)
        oil_candidates.sort(key=lambda x: abs(x[2] - 0.5))
        oil_picks = oil_candidates[:5]

        # Update or create trackers
        existing_oil = {t.label: t for t in self.oil_tracks}
        for cid, q, p in oil_picks:
            short = _shorten_oil(q)
            if short in existing_oil:
                existing_oil[short].add(p, ts)
            else:
                h = PriceHistory(short)
                h.add(p, ts)
                self.oil_tracks.append(h)
                existing_oil[short] = h
        # Prune dead tracks
        active_labels = {_shorten_oil(q) for _, q, _ in oil_picks}
        self.oil_tracks = [
            t for t in self.oil_tracks
            if t.label in active_labels or len(t.prices) > 1
        ]

        # ── Iran war chart: pick key conflict contracts ──────────────
        war_candidates: list[tuple[str, str, float]] = []
        for cid, q in self.questions.items():
            p = self.contracts.get(cid)
            if p is None:
                continue
            if IRAN_WAR_RE.search(q):
                war_candidates.append((cid, q, p))

        war_candidates.sort(key=lambda x: abs(x[2] - 0.5))
        war_picks = war_candidates[:5]

        existing_war = {t.label: t for t in self.war_tracks}
        for cid, q, p in war_picks:
            short = _shorten_war(q)
            if short in existing_war:
                existing_war[short].add(p, ts)
            else:
                h = PriceHistory(short)
                h.add(p, ts)
                self.war_tracks.append(h)
                existing_war[short] = h
        active_labels = {_shorten_war(q) for _, q, _ in war_picks}
        self.war_tracks = [
            t for t in self.war_tracks
            if t.label in active_labels or len(t.prices) > 1
        ]


def _shorten_oil(q: str) -> str:
    """Shorten oil contract question for chart label."""
    m = re.search(
        r"Crude Oil.*?(settle|hit).*?(\$\d+).*?(March|June|April)",
        q, re.IGNORECASE,
    )
    if m:
        action = ">" if "settle" in m.group(1).lower() else "hit"
        return f"CL {action} {m.group(2)} {m.group(3)[:3]}"
    return q[:25]


def _shorten_war(q: str) -> str:
    """Shorten Iran war contract question for chart label."""
    q = q.replace("Will ", "").replace("the ", "")
    q = re.sub(r"by (?:end of |)(\w+ \d+).*", r"\1", q)
    q = re.sub(r"before (\w+).*", r"\1", q)
    q = re.sub(r", 2026\??", "", q)
    return q[:30]


# ── Panel builders ───────────────────────────────────────────────────

def build_header(state: DashboardState) -> Panel:
    title = Text()
    title.append("  THE RADIANT SEER  ", style="bold white on blue")
    title.append("  ", style="")
    title.append("Iran & Oil Dashboard", style=DIM)

    status_parts = Text()
    if state.collecting:
        status_parts.append(" COLLECTING ", style="bold white on cyan")
    elif state.last_collect:
        status_parts.append(" IDLE ", style=f"bold white on {DIM}")

    status_parts.append(f"  Cycle #{state.cycle_count}", style=ACCENT)
    status_parts.append(
        f"  |  Snapshots: {state.snapshots_on_disk}", style="white"
    )

    if state.last_collect:
        ago = (datetime.now() - state.last_collect).total_seconds()
        status_parts.append(f"  |  Last: {ago:.0f}s ago", style=DIM)

    if state.next_collect:
        remaining = max(
            0, (state.next_collect - datetime.now()).total_seconds()
        )
        m, s = divmod(int(remaining), 60)
        status_parts.append(f"  |  Next: {m}m{s:02d}s", style=DIM)

    if state.collect_times:
        avg = sum(state.collect_times[-5:]) / len(state.collect_times[-5:])
        status_parts.append(f"  |  Avg: {avg:.1f}s/cycle", style=DIM)

    return Panel(
        Group(Align.center(title), Align.center(status_parts)),
        style="blue",
        padding=(0, 1),
    )


def build_scrapers_panel(state: DashboardState) -> Panel:
    table = Table(
        show_header=True, header_style="bold", box=None,
        pad_edge=False, expand=True,
    )
    table.add_column("Source", style="bold", ratio=2)
    table.add_column("Status", justify="center", ratio=1)
    table.add_column("Items", justify="right", ratio=1)

    def status_dot(ok: bool) -> Text:
        return Text(
            "  OK " if ok else " ERR ",
            style=f"bold white on {GOOD if ok else BAD}",
        )

    table.add_row(
        f"RSS ({len(DEFAULT_FEEDS)} feeds)", status_dot(state.rss_ok), str(state.rss_count)
    )
    table.add_row(
        "Reddit", status_dot(state.reddit_ok), str(state.reddit_count)
    )
    table.add_row(
        "FRED (Macro)", status_dot(state.fred_ok), str(state.fred_count)
    )
    table.add_row(
        "Polymarket", status_dot(state.poly_ok), str(state.poly_count)
    )
    table.add_row(
        "Truth Social", status_dot(state.truth_ok), str(state.truth_count)
    )

    return Panel(table, title="[bold]Scrapers[/bold]", border_style=ACCENT)


def build_headlines_panel(state: DashboardState) -> Panel:
    if not state.headlines:
        content = Text("No headlines yet", style=DIM)
    else:
        lines = []
        new_set = set(state.new_headlines)

        # Show new headlines first, then recent seen ones
        display = []
        for h in state.new_headlines[:10]:
            display.append((h, True))
        slots_left = 10 - len(display)
        if slots_left > 0:
            for h in state.headlines:
                if h not in new_set and len(display) < 10:
                    display.append((h, False))

        for h, is_new in display:
            t = Text()
            if is_new:
                t.append(" NEW ", style="bold white on green")
                t.append(" ", style="")
            else:
                t.append("     ", style="")
            t.append(h[:75], style="white" if is_new else DIM)
            lines.append(t)

        summary = Text()
        summary.append(
            f"  {len(state.new_headlines)} new this cycle "
            f"| {len(state.headlines)} total "
            f"| {state.total_unique_headlines} unique all-time",
            style=DIM,
        )
        lines.append(summary)
        content = Group(*lines)

    return Panel(
        content, title="[bold]Latest Headlines[/bold]", border_style=ACCENT
    )


def build_macro_panel(state: DashboardState) -> Panel:
    if not state.macro_values:
        return Panel(
            Text("Waiting for FRED data...", style=DIM),
            title="[bold]Macro Indicators[/bold]",
            border_style=ACCENT,
        )

    table = Table(show_header=False, box=None, pad_edge=False, expand=True)
    table.add_column("Indicator", style="bold", ratio=3)
    table.add_column("Value", justify="right", ratio=2)

    for sid in FRED_SERIES:
        val = state.macro_values.get(sid, 0.0)
        name = FRED_SHORT.get(sid, sid)
        if abs(val) > 1000:
            formatted = f"{val:,.0f}"
        elif abs(val) > 10:
            formatted = f"{val:.1f}"
        else:
            formatted = f"{val:.2f}"

        style = "white"
        if sid in ("DFF", "DGS10", "DGS2"):
            style = WARN if val > 4.0 else GOOD
        elif sid == "UNRATE":
            style = BAD if val > 5.0 else GOOD
        elif sid == "UMCSENT":
            style = BAD if val < 60 else GOOD

        table.add_row(name, Text(formatted, style=style))

    return Panel(
        table, title="[bold]Macro Indicators[/bold]", border_style=ACCENT
    )


def build_sentiment_panel(state: DashboardState) -> Panel:
    val = state.sentiment
    half = 25  # Half-width of the bar (each side of center)
    filled = int(abs(val) * half)

    if val >= 0.3:
        label, color = "Bullish", GOOD
    elif val >= 0:
        label, color = "Neutral+", "white"
    elif val >= -0.3:
        label, color = "Neutral-", WARN
    else:
        label, color = "Bearish", BAD

    # Bar: [  negative side  |  positive side  ]
    # Center "|" is always at position `half`
    bar = Text()
    bar.append(" Bearish ", style=DIM)
    bar.append("[", style=DIM)
    if val >= 0:
        # Left side empty, right side filled
        bar.append("-" * half, style=DIM)
        bar.append("|", style="bold white")
        bar.append("=" * filled, style=color)
        bar.append("-" * (half - filled), style=DIM)
    else:
        # Left side filled (from center leftward), right side empty
        bar.append("-" * (half - filled), style=DIM)
        bar.append("=" * filled, style=color)
        bar.append("|", style="bold white")
        bar.append("-" * half, style=DIM)
    bar.append("]", style=DIM)
    bar.append(" Bullish ", style=DIM)

    content = Group(
        Align.center(bar),
        Align.center(Text(f"{label}  {val:+.3f}", style=f"bold {color}")),
    )
    return Panel(content, title="[bold]Sentiment[/bold]", border_style=ACCENT)


def _build_chart_panel(
    tracks: list[PriceHistory],
    title: str,
    color: str,
    chart_width: int = 50,
) -> Panel:
    """Build a sparkline chart panel from tracked prices."""
    if not tracks:
        return Panel(
            Text("Waiting for data...", style=DIM),
            title=f"[bold]{title}[/bold]",
            border_style=color,
        )

    lines = []
    for track in sorted(tracks, key=lambda t: -(t.current or 0)):
        cur = track.current
        chg = track.change
        if cur is None:
            continue

        # Label + current price
        row = Text()
        row.append(f"  {track.label:30s} ", style="white")
        row.append(f"{cur:5.1%} ", style=f"bold {color}")

        # Change indicator
        if chg is not None and abs(chg) > 0.001:
            arrow = "+" if chg > 0 else ""
            chg_color = GOOD if chg > 0 else BAD
            row.append(f"{arrow}{chg:.1%}", style=chg_color)
        elif chg is not None:
            row.append("  --", style=DIM)

        lines.append(row)

        # Sparkline
        spark = track.sparkline(chart_width)
        if spark:
            spark_text = Text()
            spark_text.append("  ", style="")
            spark_text.append(spark, style=color)
            n = len(track.prices)
            spark_text.append(f" ({n}pts)", style=DIM)
            lines.append(spark_text)

    if not lines:
        lines.append(Text("  No matching contracts yet", style=DIM))

    return Panel(
        Group(*lines),
        title=f"[bold]{title}[/bold]",
        border_style=color,
    )


def build_oil_chart(state: DashboardState) -> Panel:
    return _build_chart_panel(
        state.oil_tracks,
        "Crude Oil Price Contracts",
        OIL_COLOR,
    )


def build_war_chart(state: DashboardState) -> Panel:
    return _build_chart_panel(
        state.war_tracks,
        "Iran War / Conflict Contracts",
        WAR_COLOR,
    )


def build_markets_panel(state: DashboardState) -> Panel:
    if not state.contracts:
        return Panel(
            Text("No market data yet", style=DIM),
            title="[bold]Polymarket Contracts[/bold]",
            border_style=ACCENT,
        )

    table = Table(
        show_header=True, header_style="bold", box=None,
        pad_edge=False, expand=True,
    )
    table.add_column("Contract", ratio=5)
    table.add_column("Price", justify="right", ratio=1)
    table.add_column("Bar", ratio=2)

    sorted_contracts = sorted(
        state.contracts.items(), key=lambda x: -x[1]
    )
    for cid, price in sorted_contracts[:12]:
        question = state.questions.get(cid, cid[:30])
        bar_len = int(price * 20)
        bar = Text()
        bar.append("=" * bar_len, style=ACCENT)
        bar.append("-" * (20 - bar_len), style=DIM)

        price_style = (
            GOOD if price > 0.7 else (BAD if price < 0.2 else "white")
        )
        table.add_row(
            Text(question[:45], style="white"),
            Text(f"{price:.1%}", style=price_style),
            bar,
        )

    remaining = len(state.contracts) - 12
    if remaining > 0:
        table.add_row(
            Text(f"  +{remaining} more contracts", style=DIM), "", ""
        )

    return Panel(
        table, title="[bold]Polymarket Contracts[/bold]", border_style=ACCENT
    )


def build_predictions_panel(state: DashboardState, bet_size: float = 20.0) -> Panel:
    """Build the predictions/P&L panel."""
    if not state.predictions_active:
        return Panel(
            Text("No model loaded — predictions disabled", style=DIM),
            title="[bold]Predictions[/bold]",
            border_style=ACCENT,
        )

    sc = state.scorecard
    lines: list[Text | str] = []

    # Model probability
    if state.p_model_range is not None:
        lo, hi = state.p_model_range
        t = Text()
        t.append("  P_model range: ", style="bold")
        t.append(f"{lo:.3f} - {hi:.3f}", style=ACCENT)
        t.append(f"   |   ${bet_size:.0f}/bet on every contract", style=DIM)
        lines.append(t)
    elif state.p_model is not None:
        t = Text()
        t.append("  P_model: ", style="bold")
        t.append(f"{state.p_model:.3f}", style=ACCENT)
        t.append(f"   |   ${bet_size:.0f}/bet on every contract", style=DIM)
        lines.append(t)

    # This cycle
    if state.cycle_scored > 0:
        pnl_color = GOOD if state.cycle_pnl >= 0 else BAD
        t = Text()
        t.append("  This cycle: ", style="bold")
        t.append(
            f"{state.cycle_correct}/{state.cycle_scored} correct "
            f"({state.cycle_correct / state.cycle_scored:.0%})",
            style="white",
        )
        t.append(f"   P&L ", style="white")
        t.append(f"${state.cycle_pnl:+,.2f}", style=pnl_color)
        lines.append(t)
    elif state.cycle_count <= 1:
        lines.append(Text("  First cycle — scoring starts next cycle", style=DIM))

    # Cumulative
    if sc.total_scored > 0:
        cum_color = GOOD if sc.total_pnl_dollars >= 0 else BAD
        t = Text()
        t.append("  Cumulative: ", style="bold")
        t.append(
            f"{sc.total_correct}/{sc.total_scored} "
            f"({sc.accuracy:.1%})",
            style="white",
        )
        t.append(f"   P&L ", style="white")
        t.append(f"${sc.total_pnl_dollars:+,.2f}", style=f"bold {cum_color}")
        lines.append(t)

    # Edge bucket breakdown
    if sc.by_edge:
        lines.append(Text(""))
        t = Text()
        t.append("  Edge       Scored  Correct  Acc     P&L", style="bold")
        lines.append(t)
        for label, _, _ in EDGE_BUCKETS:
            if label in sc.by_edge:
                b = sc.by_edge[label]
                acc = b["correct"] / b["scored"] if b["scored"] > 0 else 0.0
                pnl = b.get("pnl_dollars", 0.0)
                pnl_color = GOOD if pnl >= 0 else BAD
                acc_color = GOOD if acc > 0.55 else (BAD if acc < 0.45 else "white")
                t = Text()
                t.append(f"  {label:>6s}  ", style="white")
                t.append(f"{b['scored']:5d}  ", style="white")
                t.append(f"{b['correct']:5d}   ", style="white")
                t.append(f"{acc:5.1%}  ", style=acc_color)
                t.append(f"${pnl:+,.2f}", style=pnl_color)
                lines.append(t)

    # Learning stats
    if state.learning_active:
        lines.append(Text(""))
        t = Text()
        t.append("  Learning: ", style="bold")
        t.append(f"buffer {state.learning_buffer_size}", style="white")
        t.append(f"  |  {state.learning_total_steps} steps", style="white")
        if state.learning_loss is not None:
            t.append(f"  |  loss {state.learning_loss:.4f}", style=ACCENT)
        lines.append(t)

    if not lines:
        lines.append(Text("  Waiting for predictions...", style=DIM))

    return Panel(
        Group(*lines),
        title=f"[bold]Predictions (${bet_size:.0f}/bet)[/bold]",
        border_style=ACCENT,
    )


def build_errors_panel(state: DashboardState) -> Panel | None:
    if not state.errors:
        return None
    lines = []
    for err in state.errors[-4:]:
        t = Text()
        t.append("  ! ", style=BAD)
        t.append(err[:90], style=WARN)
        lines.append(t)
    return Panel(
        Group(*lines), title="[bold]Warnings[/bold]", border_style=WARN
    )


def build_layout(state: DashboardState, bet_size: float = 20.0) -> Group:
    """Compose all panels into the full dashboard."""
    header = build_header(state)
    scrapers = build_scrapers_panel(state)
    sentiment = build_sentiment_panel(state)
    predictions = build_predictions_panel(state, bet_size)
    headlines = build_headlines_panel(state)
    macro = build_macro_panel(state)
    oil_chart = build_oil_chart(state)
    war_chart = build_war_chart(state)
    markets = build_markets_panel(state)

    top_row = Columns([scrapers, sentiment], expand=True, equal=True)
    chart_row = Columns([oil_chart, war_chart], expand=True, equal=True)
    mid_row = Columns([headlines, macro], expand=True, equal=True)

    parts = [header, top_row, predictions, chart_row, mid_row, markets]

    errors = build_errors_panel(state)
    if errors:
        parts.append(errors)

    return Group(*parts)


# ── Collection cycle ─────────────────────────────────────────────────

def run_collection_cycle(
    state: DashboardState, scrapers: dict, prices_only: bool = False,
) -> dict:
    """Run one collection cycle, updating state along the way.

    If prices_only is True, skip headlines/sentiment/macro and only fetch
    Polymarket prices. Used by v4 contrarian mode for fast 60s cycles.
    """
    state.collecting = True
    state.errors = []
    snapshot: dict = {"timestamp": datetime.now().isoformat()}
    headlines = []

    if prices_only:
        # Fast path: only fetch Polymarket
        snapshot["headlines"] = []
        snapshot["n_headlines"] = 0
        snapshot["sentiment"] = 0.0
        snapshot["macro_values"] = [0.0] * 12
        snapshot["macro_available"] = False
        # Jump straight to Polymarket section below
        try:
            result = scrapers["polymarket"].fetch()
            state.poly_ok = result.success
            if result.success:
                data = scrapers["polymarket"].transform(result)
                state.contracts = data.get("prices", {})
                state.questions = data.get("questions", {})
                state.liquidity = data.get("liquidity", {})
                state.poly_count = len(state.contracts)
                snapshot["contracts"] = state.contracts
                snapshot["questions"] = state.questions
                snapshot["liquidity"] = state.liquidity
                snapshot["volume"] = data.get("volume", {})
            else:
                state.poly_count = 0
                snapshot["contracts"] = {}
                snapshot["questions"] = {}
                snapshot["liquidity"] = {}
                snapshot["volume"] = {}
                state.errors.append(f"Polymarket: {result.error}")
        except Exception as e:
            state.poly_ok = False
            state.poly_count = 0
            snapshot["contracts"] = {}
            snapshot["questions"] = {}
            snapshot["liquidity"] = {}
            snapshot["volume"] = {}
            state.errors.append(f"Polymarket: {e}")
        state.collecting = False
        return snapshot

    # RSS
    try:
        result = scrapers["rss"].fetch()
        state.rss_ok = result.success
        if result.success:
            data = scrapers["rss"].transform(result)
            rss_headlines = data.get("headlines", [])
            headlines.extend(rss_headlines)
            state.rss_count = len(rss_headlines)
        else:
            state.rss_count = 0
            state.errors.append(f"RSS: {result.error}")
    except Exception as e:
        state.rss_ok = False
        state.rss_count = 0
        state.errors.append(f"RSS: {e}")

    # Reddit
    try:
        result = scrapers["reddit"].fetch()
        state.reddit_ok = result.success
        if result.success:
            data = scrapers["reddit"].transform(result)
            reddit_headlines = data.get("headlines", [])
            headlines.extend(reddit_headlines)
            state.reddit_count = len(reddit_headlines)
        else:
            state.reddit_count = 0
            state.errors.append(f"Reddit: {result.error}")
    except Exception as e:
        state.reddit_ok = False
        state.reddit_count = 0
        state.errors.append(f"Reddit: {e}")

    # Truth Social
    try:
        result = scrapers["truthsocial"].fetch()
        state.truth_ok = result.success
        if result.success:
            data = scrapers["truthsocial"].transform(result)
            truth_headlines = data.get("headlines", [])
            headlines.extend(truth_headlines)
            state.truth_count = len(truth_headlines)
        else:
            state.truth_count = 0
            if result.error:
                state.errors.append(f"Truth Social: {result.error}")
    except Exception as e:
        state.truth_ok = False
        state.truth_count = 0
        state.errors.append(f"Truth Social: {e}")

    # Track new vs seen headlines
    new_this_cycle = [h for h in headlines if h not in state._seen_headlines]
    state._seen_headlines.update(headlines)
    state.new_headlines = new_this_cycle
    state.total_unique_headlines += len(new_this_cycle)

    state.headlines = headlines
    snapshot["headlines"] = headlines
    snapshot["n_headlines"] = len(headlines)

    # Content-based sentiment (on ALL headlines, not just Reddit upvotes)
    if headlines and "sentiment_analyzer" in scrapers:
        state.sentiment = scrapers["sentiment_analyzer"].score_headlines(
            headlines
        )
    else:
        state.sentiment = 0.0
    snapshot["sentiment"] = state.sentiment

    # Embed
    if headlines:
        news_emb = scrapers["embedder"].embed_aggregate(headlines)
    else:
        news_emb = np.zeros(scrapers["embedder"].dim, dtype=np.float32)
    snapshot["news_embedding"] = news_emb.tolist()

    # FRED
    try:
        result = scrapers["fred"].fetch()
        state.fred_ok = result.success
        if result.success:
            data = scrapers["fred"].transform(result)
            macro = data["macro_values"]
            snapshot["macro_values"] = macro.tolist()
            snapshot["macro_available"] = True
            state.fred_count = int((macro != 0).sum())
            series_ids = list(FRED_SERIES.keys())
            state.macro_values = {
                sid: macro[i] for i, sid in enumerate(series_ids)
            }
        else:
            snapshot["macro_values"] = [0.0] * 12
            snapshot["macro_available"] = False
            state.fred_count = 0
            state.errors.append(f"FRED: {result.error}")
    except Exception as e:
        state.fred_ok = False
        state.fred_count = 0
        snapshot["macro_values"] = [0.0] * 12
        snapshot["macro_available"] = False
        state.errors.append(f"FRED: {e}")

    # Polymarket
    try:
        result = scrapers["polymarket"].fetch()
        state.poly_ok = result.success
        if result.success:
            data = scrapers["polymarket"].transform(result)
            state.contracts = data.get("prices", {})
            state.questions = data.get("questions", {})
            state.liquidity = data.get("liquidity", {})
            state.poly_count = len(state.contracts)
            snapshot["contracts"] = state.contracts
            snapshot["questions"] = state.questions
            snapshot["liquidity"] = state.liquidity
            snapshot["volume"] = data.get("volume", {})
        else:
            state.poly_count = 0
            snapshot["contracts"] = {}
            snapshot["questions"] = {}
            snapshot["liquidity"] = {}
            snapshot["volume"] = {}
            state.errors.append(f"Polymarket: {result.error}")
    except Exception as e:
        state.poly_ok = False
        state.poly_count = 0
        snapshot["contracts"] = {}
        snapshot["questions"] = {}
        snapshot["liquidity"] = {}
        snapshot["volume"] = {}
        state.errors.append(f"Polymarket: {e}")

    # Update chart trackers
    state.update_charts()

    state.collecting = False
    return snapshot


def save_snapshot(snapshot: dict, output_dir: Path) -> Path:
    """Save snapshot without bulky repeated data.

    Strips embeddings, raw headline text (stored in headlines.jsonl),
    and questions (mostly static) to keep per-cycle snapshots small.
    """
    from radiant_seer.data_swarm.collector import append_headlines

    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"snapshot_{ts_str}.json"

    # Append new headlines to deduplicated store
    headlines = snapshot.get("headlines") or snapshot.get("headline_texts") or []
    if headlines:
        data_dir = output_dir.parent
        append_headlines(headlines, data_dir / "headlines.jsonl")

    # Strip large/repeated data
    skip = (
        "headline_embeddings", "news_embedding", "headline_timestamps",
        "headlines", "headline_texts", "questions",
    )
    slim = {k: v for k, v in snapshot.items() if k not in skip}
    slim["n_headlines"] = len(headlines)

    with open(path, "w") as f:
        json.dump(slim, f, indent=2)
    return path


def count_snapshots(output_dir: Path) -> int:
    return len(list(output_dir.glob("snapshot_*.json")))


def run_predictions(
    state: DashboardState,
    snapshot: dict,
    encoder: MultimodalEncoder,
    decoder: OutcomeDecoder,
    device: str,
    bet_size: float,
    scan_log_path: Path,
    learner: OnlineLearner | None = None,
    contract_decoder: ContractDecoder | None = None,
    relevance_router: RelevanceRouter | None = None,
    learner_v2: OnlineLearnerV2 | None = None,
    embedder: NewsEmbedder | None = None,
    contract_emb_cache: dict[str, np.ndarray] | None = None,
    kelly_sizer: "KellySizer | None" = None,
    kelly_bankroll: float = 10000.0,
    contrarian_mode: bool = False,
    contrarian_threshold: float = 0.20,
) -> None:
    """Run fast predictions on all contracts, score previous, learn.

    If contract_decoder + relevance_router are provided, uses per-contract
    predictions. Otherwise falls back to the single-p_model approach.

    If kelly_sizer is provided, uses Kelly sizing per prediction instead of
    flat bet_size. Predictions below the Kelly min_edge are skipped entirely.

    If contrarian_mode is True, skips the model entirely and bets against
    extreme prices: BUY when p_market <= threshold, SELL when >= (1-threshold).
    Uses p_model=0.50 (assume market is overconfident on extreme contracts).
    """
    contracts = state.contracts
    liquidity = state.liquidity
    if not contracts:
        return

    # Build state tensors from snapshot
    news_emb = snapshot.get("news_embedding", [0.0] * 384)
    macro_vals = snapshot.get("macro_values", [0.0] * 12)
    sentiment_val = snapshot.get("sentiment", 0.0)

    # Check if we have headline-level data for per-contract predictions
    headline_emb_list = snapshot.get("headline_embeddings")
    headline_texts = snapshot.get("headline_texts", [])
    headline_timestamps = snapshot.get("headline_timestamps", [])
    use_per_contract = (
        not contrarian_mode
        and contract_decoder is not None
        and relevance_router is not None
        and headline_emb_list is not None
        and len(headline_texts) > 0
        and embedder is not None
    )

    if contrarian_mode:
        # V4: skip encoder/decoder entirely, just need prices for scoring + contrarian
        pass
    elif use_per_contract:
        # Per-contract prediction path
        h_embs_t = torch.tensor(
            headline_emb_list if isinstance(headline_emb_list, list)
            else headline_emb_list.tolist(),
            dtype=torch.float32,
        ).to(device)
        macro_t = torch.tensor([macro_vals], dtype=torch.float32).to(device)
        sent_t = torch.tensor(
            [[sentiment_val]], dtype=torch.float32
        ).to(device)

        with torch.no_grad():
            context_z, headline_tokens = encoder.forward_with_headlines(
                h_embs_t.unsqueeze(0), macro_t, sent_t,
            )
    else:
        # Fallback: single p_model
        with torch.no_grad():
            news = torch.tensor([news_emb], dtype=torch.float32).to(device)
            macro = torch.tensor([macro_vals], dtype=torch.float32).to(device)
            sentiment = torch.tensor(
                [[sentiment_val]], dtype=torch.float32
            ).to(device)
            z = encoder(news, macro, sentiment).squeeze(0)
            p_model = decoder(z.unsqueeze(0)).item()
        state.p_model = p_model

    ts = datetime.now().isoformat()

    # 1. Score previous predictions
    scored: list[Prediction] = []
    for cid, pred in state.pending_predictions.items():
        if cid not in contracts:
            continue
        p_now = contracts[cid]
        price_move = p_now - pred.p_market
        direction_sign = 1.0 if pred.direction == "BUY" else -1.0
        profit = price_move * direction_sign

        if pred.direction == "BUY":
            entry_price = pred.p_market
        else:
            entry_price = 1.0 - pred.p_market
        this_bet = pred.bet_size
        pnl_dollars = (
            this_bet * price_move * direction_sign / entry_price
            if entry_price > 0
            else 0.0
        )

        pred.scored = True
        pred.score_timestamp = ts
        pred.p_market_after = round(p_now, 6)
        pred.price_move = round(price_move, 6)
        pred.correct = profit > 0
        pred.profit_pct = round(profit, 6)
        pred.pnl_dollars = round(pnl_dollars, 2)

        state.scorecard.record_score(pred)
        _append_scan_log(scan_log_path, {
            "type": "score",
            "prediction_timestamp": pred.timestamp,
            "score_timestamp": ts,
            "contract_id": pred.contract_id,
            "question": pred.question,
            "p_model": pred.p_model,
            "p_market": pred.p_market,
            "p_market_after": pred.p_market_after,
            "edge": pred.edge,
            "price_move": pred.price_move,
            "direction": pred.direction,
            "correct": pred.correct,
            "profit_pct": pred.profit_pct,
            "pnl_dollars": pred.pnl_dollars,
            "bet_size": this_bet,
        })
        scored.append(pred)

    state.cycle_scored = len(scored)
    state.cycle_correct = sum(1 for s in scored if s.correct)
    state.cycle_pnl = sum(s.pnl_dollars or 0.0 for s in scored)

    # 2. Learn from scored predictions
    if use_per_contract and learner_v2 is not None and scored and state.last_snapshot_data is not None:
        # V2 learning: per-contract
        prev = state.last_snapshot_data
        prev_cycle_id = prev.get("cycle_id")
        if prev_cycle_id is not None:
            for s in scored:
                if s.p_market_after is not None:
                    # Get contract embedding from cache
                    q = s.question
                    if contract_emb_cache is not None and q in contract_emb_cache:
                        c_emb = contract_emb_cache[q]
                    elif embedder is not None:
                        c_emb = embedder.embed([q])[0]
                    else:
                        continue
                    learner_v2.record_outcome(
                        cycle_id=prev_cycle_id,
                        contract_emb=c_emb,
                        contract_text=q,
                        target=s.p_market_after,
                        direction=s.direction,
                        pnl_dollars=s.pnl_dollars or 0.0,
                        p_market=s.p_market,
                        liquidity=liquidity.get(s.contract_id, 0.0),
                    )
            loss = learner_v2.learn_step()
            state.learning_loss = loss
            state.learning_buffer_size = learner_v2.buffer_size
            state.learning_total_steps = learner_v2.total_steps
    elif learner is not None and scored and state.last_snapshot_data is not None:
        # V1 learning: global
        prev = state.last_snapshot_data
        outcomes = [s.p_market_after for s in scored if s.p_market_after is not None]
        if outcomes:
            learner.record_outcomes(
                news_emb=prev["news_embedding"],
                macro=prev["macro_values"],
                sentiment=prev["sentiment"],
                outcomes=outcomes,
            )
            loss = learner.learn_step()
            state.learning_loss = loss
            state.learning_buffer_size = learner.buffer_size
            state.learning_total_steps = learner.total_steps

    # 3. Make new predictions for all contracts
    new_pending: dict[str, Prediction] = {}
    p_models_this_cycle: list[float] = []

    # ── V4: Contrarian mode — no model, just bet against extremes ──
    if contrarian_mode:
        min_liq = config.scan_min_liquidity
        ts = datetime.now().isoformat()
        for cid, p_market in contracts.items():
            if p_market <= 0.01 or p_market >= 0.99:
                continue
            # Only bet on extreme contracts
            if p_market > contrarian_threshold and p_market < (1 - contrarian_threshold):
                continue  # skip mid-range
            # Skip illiquid markets — contrarian only works on active markets
            # with real price discovery (high liquidity = real money behind the price)
            if liquidity.get(cid, 0) < 500_000:
                continue
            # Contrarian: assume true probability is 0.50
            p_model = 0.50
            edge = p_model - p_market
            direction = "BUY" if edge > 0 else "SELL"
            pred = Prediction(
                timestamp=ts,
                contract_id=cid,
                question=state.questions.get(cid, ""),
                p_model=round(p_model, 6),
                p_market=round(p_market, 6),
                edge=round(edge, 6),
                direction=direction,
            )
            if kelly_sizer is not None:
                pos = kelly_sizer.compute(p_model=p_model, p_market=p_market)
                if pos.fraction <= 0:
                    continue
                pred.bet_size = round(kelly_bankroll * pos.fraction, 2)
            else:
                pred.bet_size = bet_size
            new_pending[cid] = pred
            p_models_this_cycle.append(p_model)
            state.scorecard.record_prediction()
            _append_scan_log(scan_log_path, {
                "type": "prediction",
                "timestamp": ts,
                "contract_id": cid,
                "question": pred.question,
                "p_model": pred.p_model,
                "p_market": pred.p_market,
                "edge": pred.edge,
                "direction": pred.direction,
                "bet_size": pred.bet_size,
            })
        state.pending_predictions = new_pending
        if p_models_this_cycle:
            state.p_model_range = (min(p_models_this_cycle), max(p_models_this_cycle))
        return

    # Pre-compute headline tags ONCE (not per contract)
    cached_headline_tags = None
    if use_per_contract:
        cached_headline_tags = relevance_router.cache_headline_tags(
            headline_texts
        )

    for cid, p_market in contracts.items():
        if p_market <= 0.01 or p_market >= 0.99:
            continue

        if use_per_contract:
            # Per-contract prediction
            question = state.questions.get(cid, "")
            # Cache or compute contract embedding
            if contract_emb_cache is not None and question in contract_emb_cache:
                c_emb_np = contract_emb_cache[question]
            else:
                c_emb_np = embedder.embed([question])[0]
                if contract_emb_cache is not None:
                    contract_emb_cache[question] = c_emb_np

            c_emb_t = torch.tensor(
                c_emb_np, dtype=torch.float32
            ).to(device)

            # Compute relevance weights (headline tags pre-cached)
            weights = relevance_router.compute_weights(
                headline_embs=h_embs_t,
                headline_timestamps=headline_timestamps,
                contract_emb=c_emb_t,
                headline_texts=headline_texts,
                contract_text=question,
                headline_tags=cached_headline_tags,
            )

            with torch.no_grad():
                if isinstance(contract_decoder, ContractDecoderV2):
                    # V2: pass market context
                    hist = state.contract_history.get(
                        cid, ContractHistory()
                    )
                    mkt_ctx = torch.tensor(
                        [[p_market, hist.last_p_model,
                          hist.price_move, hist.correct]],
                        dtype=torch.float32,
                    ).to(device)
                    p_model_contract = contract_decoder(
                        context_z,
                        headline_tokens,
                        c_emb_t.unsqueeze(0),
                        weights.unsqueeze(0),
                        market_context=mkt_ctx,
                    ).item()
                else:
                    # V1: predict from scratch
                    p_model_contract = contract_decoder(
                        context_z,
                        headline_tokens,
                        c_emb_t.unsqueeze(0),
                        weights.unsqueeze(0),
                    ).item()

            p_models_this_cycle.append(p_model_contract)
            edge = p_model_contract - p_market

            # Update per-contract history for next cycle
            prev_hist = state.contract_history.get(cid, ContractHistory())
            # Was the PREVIOUS prediction's direction correct?
            # prev predicted BUY (p_model > p_market) and price went up → correct
            # prev predicted SELL (p_model < p_market) and price went down → correct
            prev_edge = prev_hist.last_p_model - prev_hist.last_p_market
            actual_move = p_market - prev_hist.last_p_market
            if prev_hist.last_p_market == 0.5:
                prev_correct = 0.5  # unknown (first time seeing this contract)
            elif actual_move == 0:
                prev_correct = 0.5  # no movement, inconclusive
            else:
                prev_correct = 1.0 if prev_edge * actual_move > 0 else 0.0
            state.contract_history[cid] = ContractHistory(
                last_p_model=p_model_contract,
                last_p_market=p_market,
                price_move=p_market - prev_hist.last_p_market,
                correct=prev_correct,
            )

            direction = "BUY" if edge > 0 else "SELL"
            pred = Prediction(
                timestamp=ts,
                contract_id=cid,
                question=question,
                p_model=round(p_model_contract, 6),
                p_market=round(p_market, 6),
                edge=round(edge, 6),
                direction=direction,
            )
        else:
            # Fallback: single p_model for all
            edge = p_model - p_market
            direction = "BUY" if edge > 0 else "SELL"
            pred = Prediction(
                timestamp=ts,
                contract_id=cid,
                question=state.questions.get(cid, ""),
                p_model=round(p_model, 6),
                p_market=round(p_market, 6),
                edge=round(edge, 6),
                direction=direction,
            )

        # Kelly sizing: compute per-prediction bet, skip if below min edge
        if kelly_sizer is not None:
            pos = kelly_sizer.compute(
                p_model=pred.p_model,
                p_market=pred.p_market,
            )
            if pos.fraction <= 0:
                continue  # skip — edge too small for Kelly
            pred.bet_size = round(kelly_bankroll * pos.fraction, 2)
        else:
            pred.bet_size = bet_size

        new_pending[cid] = pred
        state.scorecard.record_prediction()
        _append_scan_log(scan_log_path, {
            "type": "prediction",
            "timestamp": ts,
            "contract_id": cid,
            "question": pred.question,
            "p_model": pred.p_model,
            "p_market": pred.p_market,
            "edge": pred.edge,
            "direction": pred.direction,
            "bet_size": pred.bet_size,
        })

    state.pending_predictions = new_pending

    # Update p_model display
    if use_per_contract and p_models_this_cycle:
        state.p_model_range = (
            min(p_models_this_cycle),
            max(p_models_this_cycle),
        )
        state.p_model = None
    elif not use_per_contract:
        state.p_model_range = None

    # 4. Store this cycle's state for next cycle's learning
    cycle_state: dict = {
        "news_embedding": news_emb,
        "macro_values": macro_vals,
        "sentiment": sentiment_val,
    }
    if use_per_contract and learner_v2 is not None:
        cycle_id = learner_v2.record_cycle_data(
            headline_embs=headline_emb_list,
            headline_timestamps=headline_timestamps,
            headline_texts=headline_texts,
            macro=macro_vals,
            sentiment=sentiment_val,
        )
        cycle_state["cycle_id"] = cycle_id
    state.last_snapshot_data = cycle_state


def _append_scan_log(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def _load_models(
    config: SeerConfig,
) -> tuple[MultimodalEncoder, OutcomeDecoder] | None:
    """Try to load encoder + decoder. Returns None if no weights found."""
    model_dir = config.project_root / "data" / "models"

    encoder = MultimodalEncoder(
        news_dim=config.news_embedding_dim,
        macro_dim=config.macro_feature_count,
        latent_dim=config.latent_dim,
    ).to(config.device)

    decoder = OutcomeDecoder(latent_dim=config.latent_dim).to(config.device)

    # Load encoder weights
    encoder_loaded = False
    for name in ["encoder_finetuned.pt", "encoder_vicreg.pt"]:
        path = model_dir / name
        if path.exists():
            encoder.load_state_dict(torch.load(path, weights_only=True))
            encoder_loaded = True
            break

    # Load decoder weights
    dec_path = model_dir / "outcome_decoder.pt"
    decoder_loaded = dec_path.exists()
    if decoder_loaded:
        decoder.load_state_dict(torch.load(dec_path, weights_only=True))

    if not encoder_loaded or not decoder_loaded:
        return None

    encoder.eval()
    decoder.eval()
    return encoder, decoder


def _load_contract_models(
    config: SeerConfig,
    cd_path: Path | None = None,
    rs_path: Path | None = None,
    use_v2: bool = False,
) -> tuple[ContractDecoder | ContractDecoderV2, LearnedRelevanceScorer] | None:
    """Try to load contract decoder + relevance scorer weights.

    Returns None if no weights found (models still get created in main(),
    they just start with random weights).
    """
    model_dir = config.project_root / "data" / "models"
    cd_path = cd_path or (model_dir / "contract_decoder.pt")
    rs_path = rs_path or (model_dir / "relevance_scorer.pt")

    if use_v2:
        contract_decoder = ContractDecoderV2(
            latent_dim=config.latent_dim,
            contract_emb_dim=config.news_embedding_dim,
        ).to(config.device)
    else:
        contract_decoder = ContractDecoder(
            latent_dim=config.latent_dim,
            contract_emb_dim=config.news_embedding_dim,
        ).to(config.device)

    relevance_scorer = LearnedRelevanceScorer(
        emb_dim=config.news_embedding_dim,
    ).to(config.device)

    cd_loaded = False
    if cd_path.exists():
        contract_decoder.load_state_dict(
            torch.load(cd_path, weights_only=True)
        )
        cd_loaded = True

    rs_loaded = False
    if rs_path.exists():
        relevance_scorer.load_state_dict(
            torch.load(rs_path, weights_only=True)
        )
        rs_loaded = True

    contract_decoder.eval()
    relevance_scorer.eval()

    if cd_loaded or rs_loaded:
        return contract_decoder, relevance_scorer
    return None


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Radiant Seer Dashboard")
    parser.add_argument(
        "--interval", type=int, default=300,
        help="Collection interval in seconds (default: 300 = 5min)",
    )
    parser.add_argument(
        "--tag", type=str, default="",
        help="Tag for save files (e.g., 'v1' or 'v2'). "
        "Allows running multiple dashboards simultaneously.",
    )
    parser.add_argument(
        "--decoder-version", type=int, default=1, choices=[1, 2, 3, 4],
        help="Decoder version: 1=from scratch, 2=anchored on market price, "
        "3=v1 decoder + Kelly sizing, "
        "4=contrarian (no model, bet against extreme prices on ALL contracts)",
    )
    args = parser.parse_args()

    config = SeerConfig()
    bet_size = config.scan_bet_size
    tag = f"_{args.tag}" if args.tag else ""
    scan_log_path = config.data_dir / f"scan_log{tag}.jsonl"

    output_dir = (
        Path(__file__).resolve().parent.parent / "data" / "snapshots"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    embedder = NewsEmbedder()
    scrapers = {
        "rss": RssScraper(),
        "reddit": RedditScraper(),
        "truthsocial": TruthSocialScraper(),
        "fred": FredScraper(api_key=os.environ.get("FRED_API_KEY", "")),
        "polymarket": PolymarketScraper(apply_filter=False),
        "embedder": embedder,
        "sentiment_analyzer": HeadlineSentimentAnalyzer(embedder),
    }

    # Load models for predictions + learning
    models = _load_models(config)
    encoder: MultimodalEncoder | None = None
    decoder: OutcomeDecoder | None = None
    learner: OnlineLearner | None = None
    decoder_save_path = config.project_root / "data" / "models" / "outcome_decoder.pt"

    # Per-contract models — tagged save paths for A/B testing
    contract_decoder: ContractDecoder | ContractDecoderV2 | None = None
    relevance_router: RelevanceRouter | None = None
    learner_v2: OnlineLearnerV2 | None = None
    contract_emb_cache: dict[str, np.ndarray] = {}
    models_dir = config.project_root / "data" / "models"
    cd_save_path = models_dir / f"contract_decoder{tag}.pt"
    rs_save_path = models_dir / f"relevance_scorer{tag}.pt"
    buffer_save_path = models_dir / f"replay_buffer{tag}.pt"

    # Kelly sizer for v3/v4 (selective, sized bets)
    kelly_sizer: KellySizer | None = None
    kelly_bankroll = 10000.0
    contrarian_mode = False
    if args.decoder_version in (3, 4):
        kelly_sizer = KellySizer(
            kelly_fraction=config.kelly_fraction,
            max_risk=config.max_portfolio_risk,
            min_edge=config.scan_min_edge,
        )
    if args.decoder_version == 4:
        contrarian_mode = True

    state = DashboardState()
    state.snapshots_on_disk = count_snapshots(output_dir)

    if contrarian_mode:
        state.predictions_active = True
        state.learning_active = False

    if models is not None:
        encoder, decoder = models
        state.predictions_active = True
        learner = OnlineLearner(
            encoder=encoder,
            decoder=decoder,
            device=config.device,
        )
        state.learning_active = True

        # Set up per-contract models
        domain_graph = CausalDomainGraph()
        relevance_scorer = LearnedRelevanceScorer(
            emb_dim=config.news_embedding_dim,
        ).to(config.device)

        # v3 uses v1 decoder architecture + Kelly sizing
        use_v2 = args.decoder_version == 2
        if use_v2:
            contract_decoder = ContractDecoderV2(
                latent_dim=config.latent_dim,
                contract_emb_dim=config.news_embedding_dim,
            ).to(config.device)
        else:
            contract_decoder = ContractDecoder(
                latent_dim=config.latent_dim,
                contract_emb_dim=config.news_embedding_dim,
            ).to(config.device)

        # Load weights if they exist (tagged paths)
        contract_models = _load_contract_models(
            config, cd_path=cd_save_path, rs_path=rs_save_path,
            use_v2=use_v2,
        )
        if contract_models is not None:
            contract_decoder, relevance_scorer = contract_models

        relevance_router = RelevanceRouter(
            domain_graph=domain_graph,
            learned_scorer=relevance_scorer,
        )
        learner_v2 = OnlineLearnerV2(
            encoder=encoder,
            contract_decoder=contract_decoder,
            relevance_scorer=relevance_scorer,
            relevance_router=relevance_router,
            device=config.device,
        )
        saved_scorecard = learner_v2.load_buffer(buffer_save_path)
        if saved_scorecard is not None:
            state.scorecard = Scorecard.from_dict(saved_scorecard)
            state.learning_buffer_size = learner_v2.buffer_size
            state.learning_total_steps = learner_v2.total_steps

    console = Console()

    console.print(
        Panel(
            Align.center(
                Text(
                    "Starting Radiant Seer Dashboard...",
                    style=TITLE_STYLE,
                )
            ),
            style="blue",
        )
    )
    if state.predictions_active:
        version_str = ""
        if args.decoder_version == 4:
            version_str = " (V4: contrarian)"
        elif isinstance(contract_decoder, ContractDecoderV2):
            version_str = " (V2: market-anchored)"
        elif args.decoder_version == 3:
            version_str = " (V3: Kelly-sized)"
        elif contract_decoder is not None:
            version_str = " (V1: from scratch)"
        tag_str = f" [tag={args.tag}]" if args.tag else ""
        if kelly_sizer is not None:
            bet_str = f"Kelly sized (${kelly_bankroll:,.0f} bankroll)"
        else:
            bet_str = f"${bet_size:.0f}/bet"
        console.print(
            f"  Models loaded — predictions active{version_str}{tag_str} "
            f"({bet_str}, logging to {scan_log_path})"
        )
    else:
        console.print(
            "  No model weights found — predictions disabled. "
            "Train models first."
        )

    # Periodic checkpoint — save weights/buffer every 30 minutes
    _CHECKPOINT_INTERVAL = 30 * 60  # seconds
    _last_checkpoint = time.monotonic()

    with Live(
        build_layout(state, bet_size),
        console=console,
        refresh_per_second=2,
        screen=True,
    ) as live:
        try:
            while True:
                state.cycle_count += 1
                state.collecting = True
                live.update(build_layout(state, bet_size))

                t0 = time.monotonic()
                try:
                    snapshot = run_collection_cycle(
                        state, scrapers, prices_only=contrarian_mode,
                    )

                    # Add headline-level data to snapshot for per-contract path
                    if not contrarian_mode and state.headlines and embedder is not None:
                        h_embs, h_ts = embedder.embed_with_timestamps(
                            state.headlines
                        )
                        snapshot["headline_embeddings"] = h_embs.tolist()
                        snapshot["headline_timestamps"] = h_ts
                        snapshot["headline_texts"] = state.headlines

                    # Run predictions + learn after data collection
                    if encoder is not None and decoder is not None or contrarian_mode:
                        run_predictions(
                            state, snapshot, encoder, decoder,
                            config.device, bet_size, scan_log_path,
                            learner=learner,
                            contract_decoder=contract_decoder,
                            relevance_router=relevance_router,
                            learner_v2=learner_v2,
                            embedder=embedder,
                            contract_emb_cache=contract_emb_cache,
                            kelly_sizer=kelly_sizer,
                            kelly_bankroll=kelly_bankroll,
                            contrarian_mode=contrarian_mode,
                        )
                except Exception as e:
                    state.errors.append(f"Cycle error: {e}")

                elapsed = time.monotonic() - t0
                state.collect_times.append(elapsed)

                save_snapshot(snapshot, output_dir)
                state.snapshots_on_disk = count_snapshots(output_dir)
                state.last_collect = datetime.now()

                # Periodic checkpoint: save weights + buffer every 30 min
                if time.monotonic() - _last_checkpoint >= _CHECKPOINT_INTERVAL:
                    try:
                        if learner is not None and learner.total_steps > 0:
                            learner.save_decoder(decoder_save_path)
                        if learner_v2 is not None:
                            if learner_v2.total_steps > 0:
                                learner_v2.save_weights(cd_save_path, rs_save_path)
                            if learner_v2.buffer_size > 0:
                                learner_v2.save_buffer(
                                    buffer_save_path, scorecard=state.scorecard
                                )
                    except Exception as e:
                        state.errors.append(f"Checkpoint error: {e}")
                    _last_checkpoint = time.monotonic()

                live.update(build_layout(state, bet_size))

                # Sleep for remaining time (interval minus processing time)
                deadline = t0 + args.interval
                while time.monotonic() < deadline:
                    state.next_collect = datetime.fromtimestamp(
                        time.time() + (deadline - time.monotonic())
                    )
                    live.update(build_layout(state, bet_size))
                    time.sleep(1)

        except KeyboardInterrupt:
            pass

    # Save updated decoder weights
    if learner is not None and learner.total_steps > 0:
        learner.save_decoder(decoder_save_path)

    # Save per-contract model weights + replay buffer
    if learner_v2 is not None:
        if learner_v2.total_steps > 0:
            learner_v2.save_weights(cd_save_path, rs_save_path)
        if learner_v2.buffer_size > 0:
            learner_v2.save_buffer(buffer_save_path, scorecard=state.scorecard)

    # Print final summary
    sc = state.scorecard
    summary = (
        f"Stopped. {state.cycle_count} cycles, "
        f"{state.snapshots_on_disk} snapshots saved."
    )
    if sc.total_scored > 0:
        pnl_str = f"${sc.total_pnl_dollars:+,.2f}"
        summary += (
            f"\nPredictions: {sc.total_scored} scored, "
            f"{sc.total_correct} correct ({sc.accuracy:.1%}), "
            f"P&L {pnl_str}"
        )
    if learner is not None and learner.total_steps > 0:
        summary += (
            f"\nLearning (V1): {learner.total_steps} steps, "
            f"buffer {learner.buffer_size}, "
            f"loss {learner.avg_loss:.4f}"
            f"\nDecoder saved -> {decoder_save_path}"
        )
    if learner_v2 is not None and learner_v2.total_steps > 0:
        loss_str = f"{learner_v2.avg_loss:.4f}" if learner_v2.avg_loss is not None else "n/a"
        summary += (
            f"\nLearning (V2): {learner_v2.total_steps} steps, "
            f"buffer {learner_v2.buffer_size}, "
            f"loss {loss_str}"
            f"\nContract decoder saved -> {cd_save_path}"
            f"\nRelevance scorer saved -> {rs_save_path}"
            f"\nReplay buffer saved -> {buffer_save_path}"
        )
    console.print(Panel(summary, style="bold yellow"))


if __name__ == "__main__":
    main()
