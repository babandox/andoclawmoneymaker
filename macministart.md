# Mac Mini Setup

## Clone & Install

```bash
git clone https://github.com/babandox/andoclawmoneymaker.git
cd andoclawmoneymaker
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Environment Variables (optional)

```bash
export FRED_API_KEY="your_fred_api_key"  # for macro data, works without it
```

## Start the 4 Models

Run each in a separate terminal (or use `tmux`/`screen`):

### v1 — ContractDecoder, flat $20 bets (baseline)
```bash
python -m radiant_seer.dashboard --decoder-version 1
```

### v2 — ContractDecoderV2 (market-anchored), flat $20 bets
```bash
python -m radiant_seer.dashboard --tag v2 --decoder-version 2
```

### v3 — ContractDecoder + Kelly sizing (selective, sized bets)
```bash
python -m radiant_seer.dashboard --tag v3 --decoder-version 3
```

### v4 — Contrarian (no ML, bets against extreme prices, 60s cycles)
```bash
python -m radiant_seer.dashboard --tag v4 --decoder-version 4 --interval 60
```

## What Each Model Does

| Model | Decoder | Bet Sizing | Cycle | Description |
|-------|---------|------------|-------|-------------|
| v1 | ContractDecoder | Flat $20 | 5 min | Predicts probability from scratch per contract |
| v2 | ContractDecoderV2 | Flat $20 | 5 min | Anchored on market price, predicts delta correction |
| v3 | ContractDecoder | Kelly (up to $200) | 5 min | Same as v1 but only bets on edge >= 5%, sizes by conviction |
| v4 | None (contrarian) | Kelly (up to $200) | 60 sec | Bets against extreme prices (<=20% or >=80%), $500K+ liquidity only |

## Files Generated

Each model creates its own tagged files:

- `data/scan_log.jsonl` / `scan_log_v2.jsonl` / `scan_log_v3.jsonl` / `scan_log_v4.jsonl` — prediction + score history
- `data/models/contract_decoder{tag}.pt` — learned decoder weights (auto-saved every 30 min)
- `data/models/relevance_scorer{tag}.pt` — learned relevance weights
- `data/models/replay_buffer{tag}.pt` — training buffer

Shared files:
- `data/headlines.jsonl` — deduplicated headline store
- `data/snapshots/` — slim per-cycle market snapshots
- `data/models/encoder_*.pt`, `predictor.pt`, `ood_*.pt`, `outcome_decoder.pt` — pre-trained base weights (do not delete)

## tmux Quick Start

```bash
# Start a new tmux session
tmux new -s seer

# Split into 4 panes
tmux split-window -h
tmux split-window -v
tmux select-pane -t 0
tmux split-window -v

# In each pane, activate venv and run a model:
# Pane 0: source .venv/bin/activate && python -m radiant_seer.dashboard --decoder-version 1
# Pane 1: source .venv/bin/activate && python -m radiant_seer.dashboard --tag v2 --decoder-version 2
# Pane 2: source .venv/bin/activate && python -m radiant_seer.dashboard --tag v3 --decoder-version 3
# Pane 3: source .venv/bin/activate && python -m radiant_seer.dashboard --tag v4 --decoder-version 4 --interval 60

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t seer
```

## Clean Restart (if needed)

Delete online-learned weights and scan logs to start fresh:

```bash
rm -f data/models/contract_decoder*.pt data/models/relevance_scorer*.pt data/models/replay_buffer*.pt
rm -f data/scan_log*.jsonl
```

Never delete the base weights: `encoder_*.pt`, `predictor.pt`, `ood_*.pt`, `outcome_decoder.pt`.
