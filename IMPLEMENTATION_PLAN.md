# Radiant Seer — Full Implementation Plan

## Context

The **Radiant Seer** is a predictive engine for event markets (Polymarket/Kalshi) that identifies **misplaced lines**: gaps between what the model believes the probability should be vs. what the market is currently pricing. The core thesis: encode the state of the world into a latent vector, simulate forward with MCTS to estimate P_mcts, compare to P_market, and trade the divergence when confidence is high.

Example: if our model thinks a certain event should be at 80% probability but it's currently trading at 60% on prediction markets, that's a 20% gap (Seldon Divergence) we seek to exploit.

---

## Phase 0: Project Scaffolding

**Create the full project structure and tooling before writing any ML code.**

### Directory Structure

```
C:\Users\andre\andopred\
├── .gitignore
├── pyproject.toml                    # PEP 621, setuptools backend
├── radiant_seer/
│   ├── __init__.py
│   ├── main_seer.py                  # Orchestrator (stub initially)
│   ├── configuration/
│   │   ├── __init__.py
│   │   ├── contracts.yaml            # Tracked contract IDs & addresses
│   │   ├── risk_params.yaml          # Kelly/exposure limits
│   │   └── settings.py              # Pydantic Settings config loader
│   ├── intelligence/
│   │   ├── __init__.py
│   │   ├── multimodal_encoder.py     # News/Data -> 128-dim latent vector z
│   │   ├── causal_predictor.py       # z_t -> z_{t+1} transition function
│   │   ├── loss_functions.py         # VICReg loss implementation
│   │   └── expert_aggregator.py      # Historical similarity retrieval
│   ├── planning/
│   │   ├── __init__.py
│   │   ├── seer_mcts.py             # Monte Carlo Tree Search engine
│   │   ├── reward_module.py          # Contract payout / edge calculation
│   │   └── logic_guard.py           # Rule-based impossibility filter
│   ├── data_swarm/
│   │   ├── __init__.py
│   │   ├── scrapers/
│   │   │   ├── __init__.py
│   │   │   └── base_scraper.py       # Abstract scraper interface
│   │   ├── normalization.py          # Raw data -> state tensors
│   │   ├── vector_db.py             # ChromaDB long-term memory
│   │   └── synthetic.py             # Synthetic data generators (essential for Phases 1-3)
│   └── execution/
│       ├── __init__.py
│       ├── poly_interface.py         # Polymarket transaction manager
│       ├── kalshi_interface.py       # Kalshi REST API manager
│       └── kelly_sizing.py          # Position sizing (Kelly Criterion)
├── tests/
│   ├── __init__.py
│   ├── test_encoder.py
│   ├── test_predictor.py
│   ├── test_mcts.py
│   ├── test_kelly.py
│   └── test_logic_guard.py
├── notebooks/
│   └── .gitkeep
└── data/
    ├── synthetic/
    │   └── .gitkeep
    └── cache/
        └── .gitkeep
```

### Key Decisions

- **pyproject.toml + pip + venv** — no poetry/conda overhead for solo dev
- **PyTorch with CUDA** — `pip install torch --index-url https://download.pytorch.org/whl/cu124`
- **Pydantic Settings** for config — loads from YAML + environment variables
- **Sentence-transformer embeddings as encoder input** (not raw text) — keeps model ~5M params, fits in VRAM easily

### Dependencies (Phased Install)

**Phase 0 (Core):**
```
torch, numpy, pydantic, pydantic-settings, pyyaml, rich, loguru, pytest, ruff
```

**Phase 4 (Data):**
```
sentence-transformers, chromadb, feedparser, requests, aiohttp, pandas, fredapi
```

**Phase 5 (Execution):**
```
py-clob-client, httpx
```

### Central Configuration (`configuration/settings.py`)

```python
class SeerConfig(BaseSettings):
    latent_dim: int = 128
    news_embedding_dim: int = 384     # sentence-transformers default
    macro_feature_count: int = 12
    mcts_simulations: int = 1000
    mcts_exploration_constant: float = 1.41  # sqrt(2)
    alpha_threshold: float = 0.15     # Min divergence to trade (15%)
    confidence_threshold: float = 0.8
    max_portfolio_risk: float = 0.02  # 2% max per event
    kelly_fraction: float = 0.25     # Quarter-Kelly for safety
    device: str = "cuda"
```

### Deliverables
- [ ] Git repo initialized
- [ ] All directories and `__init__.py` files created
- [ ] `pyproject.toml` with phased dependencies
- [ ] `.gitignore` (Python, data, model weights, .env)
- [ ] `SeerConfig` class with YAML loading
- [ ] Virtual environment created, Phase 0 deps installed

---

## Phase 1: Intelligence Core — Encoder + VICReg

**Goal:** Build the foundation that maps "the noise of history" into a clean 128-dim latent state.

### 1a. Synthetic Data Generator (`data_swarm/synthetic.py`)

Must come first — can't test anything without data.

**Key class: `SyntheticCivStateGenerator`**
- Generates temporal sequences: `(news_embeddings, macro_tensors, sentiment, event_label, next_state_label)`
- News embeddings: random vectors in R^384 with controlled clustering (topic coherence)
- Macro tensors: correlated random walks for 12 features (inflation, rates, unemployment, GDP, etc.) with regime switches
- Sentiment: scalar derived from news + noise
- Event labels: categorical (rate_hike, election, conflict, legislation, black_swan)
- Temporal sequences with causal structure baked in (e.g., rate hike → unemployment shift)

**Key functions:**
```python
def generate_episode(length: int = 50) -> list[CivState]
def generate_dataset(n_episodes: int = 200, episode_length: int = 50) -> CivStateDataset
```

### 1b. VICReg Loss (`intelligence/loss_functions.py`)

Implement per Bardes et al. (2022) exactly. Three terms:

**Class: `VICRegLoss(lambda_var=25.0, mu_cov=1.0, nu_inv=25.0)`**

1. **Invariance Loss:** MSE between embeddings of paired states (temporally adjacent or augmented views of same state)
2. **Variance Loss:** Hinge loss ensuring `std(z) >= 1` along each dimension — prevents dimensional collapse
3. **Covariance Loss:** Penalizes off-diagonal elements of the covariance matrix — forces decorrelation between dimensions

Returns `(total_loss, {"variance": ..., "invariance": ..., "covariance": ...})` for monitoring.

### 1c. Multimodal Encoder (`intelligence/multimodal_encoder.py`)

**Architecture: Modality-specific projection heads + cross-attention fusion**

```python
class MultimodalEncoder(nn.Module):
    # Per-modality projection MLPs
    news_proj:      Linear(384 → 256 → 128) with GELU + LayerNorm
    macro_proj:     Linear(12 → 64 → 128)  with GELU + LayerNorm
    sentiment_proj: Linear(1 → 32 → 128)   with GELU

    # Cross-modal fusion
    fusion: 2-layer TransformerEncoder(d_model=128, nhead=8, batch_first=True)
    output_proj: Linear(128 → 128)

    def forward(news_emb, macro_tensor, sentiment) -> Tensor:  # shape: (batch, 128)
        # 1. Project each modality to 128-dim
        # 2. Stack as sequence of 3 tokens
        # 3. Fuse via Transformer
        # 4. Mean-pool to single 128-dim vector z
```

**Why this over a full Hierarchical Transformer:**
- Taking pre-computed sentence embeddings (from `sentence-transformers`) as input keeps the model ~5M params
- The "hierarchical" aspect is preserved: news (high-freq) and macro (low-freq) have separate projection paths
- Fusion happens at the end via cross-attention
- Can upgrade to raw-text transformer later if needed

**Training:**
- Pairs of `(state_t, state_t+1)` from synthetic data
- VICReg loss between `z_t` and `z_t+1` (similar but not identical for consecutive states)
- Additional VICReg on `(z_t, augmented_z_t)` for robustness

### Tests (`tests/test_encoder.py`)
- VICReg loss components compute correctly on known inputs
- Encoder output shape is `(batch, 128)` for various input shapes
- Training doesn't collapse (per-dimension std > threshold after N epochs)
- Different regimes form distinct clusters (validate in notebook with t-SNE/UMAP)

### Exit Criteria
- Encoder trains on synthetic data without collapse
- Distinct clusters for normal / crisis / black swan regimes in embedding space

### Deliverables
- [ ] `synthetic.py` with `SyntheticCivStateGenerator` and `CivStateDataset`
- [ ] `loss_functions.py` with `VICRegLoss`
- [ ] `multimodal_encoder.py` with `MultimodalEncoder`
- [ ] Training script or notebook for encoder
- [ ] `test_encoder.py` passing
- [ ] t-SNE visualization confirming regime separation

---

## Phase 2: Causal Predictor

**Goal:** Learn the one-step transition function: z_{t+1} = f(z_t, event).

### Causal Predictor (`intelligence/causal_predictor.py`)

```python
class CausalPredictor(nn.Module):
    event_embedding: nn.Embedding(num_event_types, 128)
    predictor: Sequential(
        Linear(256 → 256) + GELU + LayerNorm,
        Linear(256 → 256) + GELU + LayerNorm,
        Linear(256 → 128)
    )

    def forward(z_t: Tensor, event_id: Tensor) -> Tensor:
        event_emb = self.event_embedding(event_id)
        combined = cat([z_t, event_emb], dim=-1)  # (batch, 256)
        return self.predictor(combined)             # (batch, 128)
```

**Design decisions:**
- **MLP, not RNN** — intentional. MCTS handles temporal chaining by calling this repeatedly. The predictor is a pure one-step transition function.
- **Categorical event embeddings** initially (not continuous). Simpler, can upgrade later.
- **VICReg training:** predicted `z_{t+1}` should match encoded `z_{t+1}` from the encoder

**Training:**
```python
z_t = encoder(state_t)           # frozen encoder weights
z_t1_true = encoder(state_{t+1}) # target
z_t1_pred = predictor(z_t, event)
loss = vicreg_loss(z_t1_pred, z_t1_true)
```

Start with encoder frozen. Optionally fine-tune both end-to-end once predictor converges.

### Tests (`tests/test_predictor.py`)
- Given z_t for "normal economy" + event="rate_hike", predicted z_{t+1} shifts toward "tightening" cluster
- Different events from the same z_t produce different z_{t+1}
- Multi-step rollout (apply predictor 5x in sequence): trajectories stay in-distribution (no explosion, no collapse)

### Exit Criteria
- Predictor generates plausible multi-step trajectories on synthetic data
- Different event sequences produce measurably different terminal states

### Deliverables
- [ ] `causal_predictor.py` with `CausalPredictor`
- [ ] Training script/notebook
- [ ] `test_predictor.py` passing
- [ ] Multi-step rollout visualization

---

## Phase 3: MCTS Search Engine — "The Psychohistory Tree"

**Goal:** Search the tree of possible futures to estimate P_mcts for any contract outcome. This is the core differentiator — where misplaced lines get detected.

### 3a. MCTS Core (`planning/seer_mcts.py`)

```python
@dataclass
class MCTSNode:
    z_state: Tensor               # Latent state at this node
    event: int | None             # Event that led here
    parent: MCTSNode | None
    children: dict[int, MCTSNode] # event_id -> child
    visit_count: int = 0
    total_value: float = 0.0
    prior: float = 1.0

class SeerMCTS:
    def search(z_root, target_contract, n_simulations=1000) -> MCTSResult:
        """Run MCTS, return probability distribution over outcomes."""

    def _select(node) -> MCTSNode:
        """UCT: Q(child) + C * sqrt(ln(N_parent) / N_child)"""

    def _expand(node) -> MCTSNode:
        """Create child nodes for all possible events."""

    def _simulate(node, depth) -> float:
        """Rollout using CausalPredictor for depth steps, return reward."""

    def _backpropagate(node, value):
        """Update visit counts and values up the tree."""
```

**UCT formula:**
```
UCT(child) = (total_value / visit_count) + C * sqrt(ln(parent.visits) / child.visits)
```
C = 1.41 (sqrt(2)) by default. Balances exploitation of known-good branches vs. exploration of unlikely but high-payout scenarios.

### 3b. Reward Module (`planning/reward_module.py`)

**This is where misplaced lines get quantified.**

```python
class RewardModule:
    outcome_decoder: nn.Module  # MLP: 128 → 64 → 1 (sigmoid)

    def compute_reward(z_terminal, contract, market_price) -> float:
        P_model = outcome_decoder(z_terminal)  # What WE think probability is
        P_market = market_price                 # What the MARKET thinks
        edge = P_model - P_market               # The mispricing we exploit
        return edge
```

The `outcome_decoder` maps a terminal latent state to a probability for a specific contract outcome. Train on synthetic data initially where ground truth is known.

### 3c. Logic Guard (`planning/logic_guard.py`)

Grounding the simulation in reality. Start with 5-10 hard-coded Python rules:

```python
@dataclass
class Rule:
    name: str
    check_fn: Callable[[dict], bool]
    description: str

class LogicGuard:
    rules: list[Rule]

    def validate_transition(z_t, event, z_t1, metadata) -> ValidationResult
    def validate_trajectory(trajectory: list[MCTSNode]) -> ValidationResult
```

**Example rules:**
- Election cannot resolve before constitutionally mandated date
- Fed cannot change rates by more than 100bps in one meeting
- Effect cannot precede cause (temporal causality)
- Physical resource constraints (e.g., oil supply bounds)
- Legislative process minimum timelines

Not a formal logic engine — just Python functions. Keep it simple.

### Tests
- MCTS converges: output distribution stabilizes as simulations increase (10 → 100 → 1000)
- LogicGuard correctly rejects impossible transitions
- Reward module produces correct edge calculations
- Performance: 1000 simulations < 10 seconds on GPU

### Exit Criteria
- Stable probability estimates on synthetic scenarios
- LogicGuard filters impossible branches
- Performance target met

### Deliverables
- [ ] `seer_mcts.py` with `SeerMCTS` and `MCTSNode`
- [ ] `reward_module.py` with `RewardModule` and `outcome_decoder`
- [ ] `logic_guard.py` with `LogicGuard` and initial rules
- [ ] `test_mcts.py` and `test_logic_guard.py` passing
- [ ] Convergence analysis notebook

---

## Phase 4: Real Data Pipeline

**Goal:** Transition from synthetic to real-world data. Start pragmatic — 3-5 sources, not 50.

### 4a. Normalization (`data_swarm/normalization.py`)

```python
class StateNormalizer:
    def normalize_news(headlines: list[str], model: SentenceTransformer) -> Tensor
        # Batch-encode headlines, mean-pool to single news embedding

    def normalize_macro(macro_dict: dict[str, float]) -> Tensor
        # Z-score normalize using running statistics

    def normalize_sentiment(raw_sentiment: float) -> Tensor
        # Clip and scale to [-1, 1]

    def to_state_tensor(news, macro, sentiment) -> CivState
        # Bundle into format the encoder expects
```

### 4b. Scrapers (`data_swarm/scrapers/`)

**Start with 3-5 high-quality free/cheap sources (80/20 rule):**

| Source | File | API Key? | What It Provides |
|--------|------|----------|-----------------|
| RSS feeds (Reuters, AP, BBC, Bloomberg) | `rss_scraper.py` | No | Dozens of headlines/hour for news embeddings |
| FRED (Federal Reserve Economic Data) | `fred_scraper.py` | Free key | All macro tensors: inflation, rates, unemployment, GDP |
| Polymarket API | `polymarket_scraper.py` | Yes | Market prices, volume, order books (your target market) |
| Reddit JSON | `reddit_scraper.py` | No | Social sentiment from relevant subreddits |

**Base pattern:**
```python
class BaseScraper(ABC):
    async def fetch() -> list[RawDataPoint]
    def transform(raw) -> list[NormalizedDataPoint]
    async def run() -> list[NormalizedDataPoint]
```

**Deferred sources:** Twitter/X (expensive, unstable API), legislative trackers (niche), polling data (election-cycle dependent), the other 45+ sources

### 4c. Vector DB (`data_swarm/vector_db.py`)

**ChromaDB** — pure Python, local persistence, zero infrastructure.

```python
class SeerMemory:
    def store_resolution(event_id, z_state, outcome, metadata)
    def find_similar_states(z_query, k=5) -> list[HistoricalEvent]
```

When MCTS encounters a state, retrieve historically similar states and their actual outcomes. Grounds simulation in real history.

### 4d. Expert Aggregator (`intelligence/expert_aggregator.py`)

Minimal version — nearest-neighbor retrieval from vector DB, not full RAG:

```python
class ExpertAggregator:
    def get_expert_prior(z_current) -> Tensor
        # Retrieve similar historical states, compute weighted average outcome
        # Use as prior for MCTS
```

### Tests
- Each scraper integration test with live API calls (skip in CI)
- Normalization produces consistent tensor shapes regardless of input count
- ChromaDB store/retrieve round-trip
- Full pipeline: scrape → normalize → encode → z produces valid latent vectors

### Deliverables
- [ ] `normalization.py` with `StateNormalizer`
- [ ] `base_scraper.py` abstract class
- [ ] `rss_scraper.py`, `fred_scraper.py`, `polymarket_scraper.py`, `reddit_scraper.py`
- [ ] `vector_db.py` with `SeerMemory` (ChromaDB)
- [ ] `expert_aggregator.py` with nearest-neighbor retrieval
- [ ] Integration tests for each scraper

---

## Phase 5: Execution Layer

**Goal:** Turn alpha signals into positions. Paper trading first, always.

### 5a. Kelly Sizing (`execution/kelly_sizing.py`)

Build first — pure math, no external dependencies.

```python
class KellySizer:
    def __init__(max_risk_per_event=0.02):
        ...

    def optimal_fraction(p_model, p_market, payout_ratio=1.0) -> float:
        # Kelly: f* = (p * b - q) / b
        # For binary markets: b = (1 - p_market) / p_market

    def size_position(portfolio_value, p_model, p_market) -> PositionSize:
        # Dollar amount, capped at max_risk * portfolio_value
        # Uses QUARTER-KELLY (0.25x) — full Kelly is too aggressive with estimation error
```

**Why quarter-Kelly:** Full Kelly is theoretically optimal but assumes perfect probability estimates. With model uncertainty, full Kelly leads to ruin. Quarter-Kelly provides ~75% of the long-run growth rate with dramatically lower variance and drawdown.

### 5b. Alpha Detection + OOD Protection (in `main_seer.py`)

```python
class AlphaDetector:
    def compute_alpha(p_mcts, p_market) -> float:
        return abs(p_market - p_mcts)  # The Seldon Divergence

    def compute_confidence(z_target) -> float:
        entropy = Normal(z.mean(), z.std()).entropy().mean()
        return 1.0 / (entropy + 1e-8)

    def should_trade(alpha, confidence, config) -> bool:
        return alpha > config.alpha_threshold and confidence > config.confidence_threshold
        # Default: alpha > 15% AND confidence > 0.8

class OODDetector:
    def is_ood(z, threshold=3.0) -> bool:
        # Mahalanobis distance from training distribution center
        # If OOD ("Mule Protection"): liquidate ALL positions immediately
```

### 5c. Polymarket Interface (`execution/poly_interface.py`)

```python
class PolymarketInterface:
    def __init__(config):
        self.client = ClobClient(host="https://clob.polymarket.com", ...)

    def get_market_price(token_id) -> float
    def place_limit_order(token_id, side, price, size) -> OrderResponse
    def get_positions() -> list[Position]
    def cancel_all() -> None  # Emergency liquidation
```

### 5d. Kalshi Interface (`execution/kalshi_interface.py`)

```python
class KalshiInterface:
    # REST API via httpx
    async def get_market_price(ticker) -> float
    async def place_order(ticker, side, count, price) -> OrderResponse
    async def get_positions() -> list[Position]
```

**Defer Kalshi until Polymarket is fully working.**

### 5e. Paper Trading Wrapper

**Mandatory before real execution.** A wrapper that:
- Accepts the same interface as real exchange clients
- Logs every order decision with full context (z_state, alpha, confidence, size)
- Tracks simulated PnL
- Does NOT send any orders to exchanges

### Tests
- Kelly sizing: unit tests with known probabilities, verify formula correctness
- OOD detector: test with in-distribution and out-of-distribution synthetic vectors
- Exchange interfaces: integration tests against paper trading wrapper
- Alpha detector: verify threshold logic

### Deliverables
- [ ] `kelly_sizing.py` with `KellySizer`
- [ ] `AlphaDetector` and `OODDetector` in `main_seer.py`
- [ ] `poly_interface.py` with `PolymarketInterface`
- [ ] `kalshi_interface.py` with `KalshiInterface` (stub)
- [ ] Paper trading wrapper
- [ ] `test_kelly.py` passing

---

## Phase 6: System Orchestration

**Goal:** Wire everything together into a live monitoring loop.

### `main_seer.py` — The Main Loop

```python
class RadiantSeer:
    def __init__(config: SeerConfig):
        self.encoder = MultimodalEncoder(config)
        self.predictor = CausalPredictor(config)
        self.mcts = SeerMCTS(self.predictor, self.reward, config)
        self.kelly = KellySizer(config.max_portfolio_risk)
        self.alpha_detector = AlphaDetector()
        self.ood_detector = OODDetector(training_stats)
        self.scrapers = [RSScraper(), FREDScraper(), PolymarketScraper()]
        self.normalizer = StateNormalizer()

    async def run_cycle():
        # 1. Scrape latest data from all sources
        # 2. Normalize → state tensor
        # 3. Encode → latent z
        # 4. OOD check → emergency liquidation if out-of-distribution
        # 5. For each tracked contract:
        #    a. MCTS search → P_mcts (what we think probability is)
        #    b. Get P_market from exchange (what market is pricing)
        #    c. Compute Seldon Divergence: Δ = |P_market - P_mcts|
        #    d. Compute confidence = 1/Entropy(z_target)
        #    e. If Δ > 15% AND confidence > 0.8:
        #       → Kelly size the position (quarter-Kelly, capped at 2%)
        #       → Place order (paper or live)
        # 6. Log everything with full context

    async def run_loop(interval_minutes=15):
        # Run cycles on a timer
```

### Monitoring
- **`rich`** console dashboard showing: current latent state, tracked contracts, alpha values, positions, PnL
- **`loguru`** structured logging: every trade decision with full context for post-hoc analysis

### Deliverables
- [ ] `RadiantSeer` orchestrator class
- [ ] `run_cycle()` and `run_loop()` methods
- [ ] Rich console dashboard
- [ ] Structured logging

---

## Phase 7: Backtesting & Validation

**No real money until this phase is complete and shows positive results.**

### 7a. Historical Data Collection
- Collect historical Polymarket prices and resolution outcomes
- Collect corresponding news/macro data for the same time periods
- Store in structured format for replay

### 7b. Backtester

```python
class Backtester:
    def run(start_date, end_date) -> BacktestResult
        # Replay historical data through the full pipeline, track PnL

    def analyze(result) -> AnalysisReport
        # Sharpe ratio, max drawdown, win rate, calibration plot
```

### 7c. Calibration Analysis

**The most critical test:** Are the model's probability estimates well-calibrated? When the model says 70%, does the event happen ~70% of the time?

- Plot reliability diagrams (calibration curves)
- Compute Brier scores
- Analyze edge distribution: are we consistently finding real mispricings, or is it noise?

### Requirements Before Going Live
- [ ] Positive Sharpe ratio on backtest
- [ ] Calibration within 5% across probability buckets
- [ ] Max drawdown < 15% of paper portfolio
- [ ] At least 2 weeks of paper trading with positive results
- [ ] Manual review of every paper trade decision

---

## Deferred Features (Build Later)

| Feature | Why Defer |
|---------|-----------|
| 50+ news sources | 3-5 sources cover ~80% of signal for ~5% of effort |
| Twitter/X API | Expensive, unreliable API, limited value vs. RSS |
| Full RAG expert aggregator | Nearest-neighbor from vector DB suffices initially |
| Kalshi integration | Get Polymarket working first, add Kalshi once core works |
| Sentiment feedback loop modeling | Requires significant position sizes and market impact data |
| Hierarchical Transformer over raw text | Sentence embeddings are more practical for ~5M param budget |
| Continuous event embeddings | Categorical types work for initial prototype, upgradeable |
| Self-fulfilling prophecy loop | Relevant only at scale, not for initial deployment |
| Formal logic engine (Prolog/Z3) | 5-10 Python rule functions cover initial markets |

---

## Risk Mitigations

| Risk | Mitigation |
|------|------------|
| VICReg embedding collapse | Monitor per-dimension variance; early stop if variance drops |
| MCTS computational cost | Cap tree depth at 5-7 steps; batch GPU forward passes |
| Overfitting to synthetic data | Ensure synthetic diversity; transition to real data ASAP (Phase 4) |
| Model calibration drift | Continuous monitoring; retrain on rolling window |
| Exchange API changes | Abstract behind interface classes; isolate exchange-specific code |
| Estimation error in probabilities | Quarter-Kelly sizing (not full Kelly); hard 2% risk cap per event |
| Black swan / OOD event | Mahalanobis OOD detection → automatic full liquidation |
| 12GB VRAM limitation | Sentence embeddings (not raw text); batch MCTS rollouts |
| Windows path issues | Use `pathlib` everywhere; test with forward slashes |

---

## Implementation Order Summary

```
Phase 0 → Scaffolding, config, venv, git
Phase 1 → Synthetic data → VICReg loss → Multimodal encoder (TRAIN)
Phase 2 → Causal predictor (TRAIN)
Phase 3 → MCTS + Reward module + Logic guard
Phase 4 → Scrapers + Normalization + Vector DB + Expert aggregator
Phase 5 → Kelly sizing + Alpha detection + OOD + Exchange interfaces + Paper trading
Phase 6 → Orchestrator + Monitoring + Live loop
Phase 7 → Backtesting + Calibration + Go/No-go for real money
```

Each phase produces runnable, testable code. No phase depends on anything after it.
