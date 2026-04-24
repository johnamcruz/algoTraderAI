# AI Algo Trading Bot

![Python Tests](https://github.com/johnamcruz/algoTraderAI/actions/workflows/tests.yml/badge.svg)

## Overview

A Python algorithmic trading bot for futures markets (MNQ, MES, MGC) via the TopstepX/ProjectX platform. Uses a **CISD+OTE** (Change in State of Delivery + Optimal Trade Entry) AI strategy for entry signals, with dynamic risk-based position sizing.

**Active model:** `cisd_ote_hybrid_v7.onnx` — FFM Hybrid Transformer (see [MODEL_CONTEXT.md](MODEL_CONTEXT.md))

### Key Features

- 🧠 **CISD+OTE Strategy v7** — ICT-style zone detection with a 96-bar FFM Transformer backbone. The risk head predicts a per-trade R:R ratio that snaps to calibrated TP tiers (1.5R / 2R / 3R / 4R) — dynamic profit targets without manual tuning.
- 📊 **Real-time Data Processing** — Live tick aggregation and bar generation (1, 3, or 5 minute bars)
- 💰 **Risk-Based Position Sizing** — Size contracts dynamically from a dollar risk budget; skip signals that exceed it
- 🔬 **Backtesting Mode** — Replay historical CSV data with realistic gap-open fills and wick-based exit simulation
- 🎯 **RR Gate** — Filters low-expectancy signals; only enters when the model predicts `>= min_risk_rr` (default 2.0)
- 📋 **Per-Trade Signal Analysis** — Logs feature averages for winners vs losers after each backtest run
- ⚙️ **YAML Configuration** — Manage live and backtest configs via file; CLI args override
- 🔄 **Backtest Runner** — `backtest.py` runs predefined market-regime scenarios (bear, recovery, banking crisis, selloff, OOS) in parallel

---

## ⚠️ Disclaimer

**Trading futures involves substantial risk of loss and is not suitable for all investors.**

- Use at your own risk — past performance is not indicative of future results
- Paper/sim trade thoroughly before going live
- Only use risk capital you can afford to lose

---

## Architecture

```
algoTrader.py                    # Entry point — arg parsing, mode dispatch
├── bots/
│   ├── trading_bot.py           # Live bot: SignalR, tick aggregation, order execution
│   ├── simulation_bot.py        # Backtest bot: CSV replay, P&L tracking, signal analysis
│   └── trading_bot_base.py      # Shared logic: AI prediction loop, sizing, exits
├── strategies/
│   ├── strategy_cisd_ote_v7.py  # CISD+OTE v7: FFM Transformer backbone + risk head TP
│   ├── strategy_cisd_ote.py     # CISD+OTE v5.1: 32-feature vector, fixed TP
│   ├── strategy_base.py         # Abstract base class for strategies
│   └── strategy_factory.py      # Strategy registry
├── utils/
│   ├── bot_utils.py             # Auth, logging, tick value/size lookups
│   └── config_loader.py         # YAML config loading and validation
└── backtest.py                  # Scenario backtest runner (parallel, multi-symbol)
```

---

## Installation

### Requirements

- Python 3.10+
- TopstepX or ProjectX-compatible API access (live mode only)

### Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate          # macOS/Linux
.\venv\Scripts\activate           # Windows
```

### Install Dependencies

```bash
pip install onnxruntime pandas pandas-ta signalrcore requests numpy scikit-learn python-dateutil pyyaml futures-foundation
```

### Prepare Model Files

Place your trained model in the `models/` folder:

```
models/
├── cisd_ote_hybrid_v7.onnx       # v7 — active model (recommended)
└── cisd_ote_hybrid_v5_1.onnx     # v5.1 — available for comparison
```

---

## Usage

### Backtest Runner (Recommended)

`backtest.py` runs predefined market-regime scenarios against historical data:

```bash
# Run all scenarios (MNQ default, v7 model)
python3 backtest.py --parallel

# Run a single scenario
python3 backtest.py --scenario banking_2023

# Run on a different symbol
python3 backtest.py --symbol MGC --parallel

# Override entry confidence
python3 backtest.py --entry_conf 0.85 --parallel

# List available scenarios
python3 backtest.py --list
```

**Available scenarios:**

| Key | Period | Description |
|-----|--------|-------------|
| `bear_2022` | 2022-01-01 → 2022-10-15 | Persistent downtrend |
| `recovery_2023` | 2023-01-01 → 2023-12-31 | Post-rate-hike rebound + AI hype |
| `banking_2023` | 2023-03-01 → 2023-05-31 | SVB collapse, high-vol chop |
| `selloff_2024` | 2024-07-15 → 2024-09-15 | Sharp selloff + recovery |
| `oos_2021` | 2021-01-01 → 2021-12-31 | Out-of-sample control year |
| `recent_120d` | 2025-06-27 → 2025-10-24 | Most recent 120-day window |

**Supported symbols:** `MNQ`, `MES`, `MGC`

### Direct Backtesting — v7 (Recommended)

```bash
python algoTrader.py \
    --backtest \
    --backtest_data data/NQ_continuous_5min.csv \
    --contract CON.F.US.MNQ.M26 \
    --strategy cisd-ote7 \
    --model models/cisd_ote_hybrid_v7.onnx \
    --entry_conf 0.80 \
    --min_risk_rr 2.0 \
    --risk_amount 200
```

### Direct Backtesting — v5.1 (Legacy)

```bash
python algoTrader.py \
    --backtest \
    --backtest_data data/NQ_continuous_5min.csv \
    --contract CON.F.US.MNQ.M26 \
    --strategy cisd-ote \
    --model models/cisd_ote_hybrid_v5_1.onnx \
    --entry_conf 0.70 \
    --risk_amount 300
```

### Live Trading — v7

```bash
python algoTrader.py \
    --account TS001234SIM \
    --contract CON.F.US.MNQ.M26 \
    --username YourUsername \
    --apikey YourApiKey \
    --timeframe 5 \
    --strategy cisd-ote7 \
    --model models/cisd_ote_hybrid_v7.onnx \
    --entry_conf 0.80 \
    --min_risk_rr 2.0 \
    --risk_amount 200
```

### YAML Config (Recommended for Live)

```yaml
# configs/cisd_ote7_mnq.yaml
username: "YourUsername"
apikey: "YourApiKey"
account: "TS001234SIM"
contract: "CON.F.US.MNQ.M26"
timeframe: 5

strategy: "cisd-ote7"
model: "models/cisd_ote_hybrid_v7.onnx"

entry_conf: 0.80
min_risk_rr: 2.0
risk_amount: 200
```

```bash
python algoTrader.py --config configs/cisd_ote7_mnq.yaml
```

---

## CLI Arguments

### Core

| Argument | Description | Default |
|----------|-------------|---------|
| `--strategy` | `cisd-ote7` (v7, recommended) or `cisd-ote` (v5.1) | — |
| `--model` | Path to ONNX model file | — |
| `--contract` | Contract ID (e.g. `CON.F.US.MNQ.M26`) | — |
| `--timeframe` | Bar timeframe in minutes (1, 3, 5) | `5` |
| `--entry_conf` | Min AI confidence to enter (0.0–1.0) | `0.9` |

### Risk & Sizing

| Argument | Description | Default |
|----------|-------------|---------|
| `--risk_amount` | Max dollars to risk per trade (dynamic sizing) | None |
| `--size` | Fixed contract count (used when `--risk_amount` not set) | `1` |
| `--max_contracts` | Cap on contracts per trade | `15` |
| `--max_loss` | Stop bot when cumulative loss hits this (dollars) | `3000` |
| `--profit_target` | Stop bot when cumulative P&L reaches this (dollars) | `6000` |
| `--min_stop_atr` | Minimum stop size as ATR multiple (prevents micro stops) | `0.5` |
| `--min_stop_pts` | Minimum stop size in points (floor) | `1.0` |

### Entry Filters

| Argument | Strategy | Description | Default |
|----------|----------|-------------|---------|
| `--min_risk_rr` | `cisd-ote7` | Skip signals when model's predicted R:R is below this | `2.0` |
| `--min_vty_regime` | `cisd-ote` | Regime gate: skip entries when `atr14/atr_ma50` is below this (0.0 = off) | `0.75` |
| `--min_entry_distance` | `cisd-ote` | OTE depth gate: minimum zone penetration depth (0.0 = off) | `3.0` |
| `--high_conf_multiplier` | `cisd-ote` only | Extend profit target at ≥90% confidence (disabled for `cisd-ote7`) | `1.0` |

### Backtesting

| Argument | Description | Default |
|----------|-------------|---------|
| `--backtest` | Run in backtesting mode | — |
| `--backtest_data` | Path to OHLCV CSV file | — |
| `--start-date` | Start date (`YYYY-MM-DD`) | None |
| `--end-date` | End date (`YYYY-MM-DD`) | None |

### Live Only

| Argument | Description |
|----------|-------------|
| `--account` | TopstepX account ID |
| `--username` | TopstepX username |
| `--apikey` | TopstepX API key |

---

## Strategy Comparison

Two strategies are available. Both share the same CISD+OTE entry concept; the difference is the model architecture and how the profit target is set.

| | `cisd-ote7` (v7) | `cisd-ote` (v5.1) |
|---|---|---|
| **Model** | `cisd_ote_hybrid_v7.onnx` | `cisd_ote_hybrid_v5_1.onnx` |
| **Backbone** | 96-bar FFM Transformer | 64-bar FFM Transformer |
| **Profit target** | Dynamic — risk head predicts R:R, snapped to 1.5R/2R/3R/4R tier | Fixed 2R (4R at ≥90% conf via `--high_conf_multiplier`) |
| **Entry confidence** | 0.80 recommended | 0.70–0.85 |
| **RR gate** | `--min_risk_rr 2.0` | Not applicable |
| **Signal frequency** | ~2× more signals | Fewer signals, stricter built-in gates |
| **Session gate** | None — model self-regulates | Hard: 7am–4pm ET only |
| **Best for** | Default — dynamic TP, higher frequency | Comparison runs or markets where session gate helps |

**v7 is the recommended default.** It generates more signals and sets profit targets dynamically based on what the model predicts, rather than relying on a fixed multiplier. The `min_risk_rr=2.0` gate keeps precision high by skipping setups where the model is pessimistic about the trade's reward potential.

---

## Multi-Ticker Setup

The strategy is designed to run simultaneously across uncorrelated instruments:

| Ticker | Contract | Data | Notes |
|--------|----------|------|-------|
| MNQ | `CON.F.US.MNQ.M26` | `NQ_continuous_5min.csv` | Primary — highest signal quality |
| MES | `CON.F.US.MES.M26` | `ES_continuous_5min.csv` | Correlated to MNQ — adds trade frequency |
| MGC | `CON.F.US.MGC.M26` | `GC_continuous_5min.csv` | Uncorrelated to equities — genuine diversification |

Run each ticker in a separate terminal or process. All three use the same model.

---

## Risk Management

### Dynamic Sizing (Recommended)

Set `--risk_amount` and the bot sizes each trade to risk exactly that many dollars at the stop level:

```
contracts = floor(risk_amount / (stop_ticks × tick_value))
```

- If even 1 contract exceeds the budget, the signal is skipped
- `--min_stop_atr` and `--min_stop_pts` prevent unrealistically tight stops from inflating size
- For `cisd-ote7`, profit target is set by the risk head (tier-snapped) — `high_conf_multiplier` is always disabled

### Gap Risk

Sizing targets the stop level. If a bar opens beyond the stop (gap), the exit fills at open — actual loss may exceed the risk budget. This is realistic market behavior.

---

## Per-Trade Signal Analysis

After each backtest, a winners-vs-losers feature table is printed:

```
Feature                   Winners     Losers      Delta
--------------------------------------------------------
signal_prob                0.8234     0.7891    +0.0343
risk_rr                    2.8801     1.9123    +0.9678
entry_distance_pct         4.1200     2.3400    +1.7800
...
```

This helps identify which features separate winning trades from losing ones and can inform future gate calibration.

---

## Monitoring

```bash
# Tail logs
tail -f logs/bot_MNQ_live_YYYYMMDD.log

# Filter entries/exits
grep "ENTRY\|EXIT" logs/bot_MNQ_live_YYYYMMDD.log

# Check errors
grep "ERROR" logs/bot_MNQ_live_YYYYMMDD.log
```

---

## Known Issues & Limitations

### Persistent Trend Blindness (v7)

v7's 96-bar context window covers ~8 hours. In sustained multi-week downtrends (e.g. the 2022 bear market), the model fires bullish CISD setups because local structure looks valid — it has no awareness that the daily or weekly trend is bearish. This is the primary driver of underperformance in `bear_2022`. A future version should add a daily HTF structure feature to address this.

### Session Gate Removed in v7

v5.1 enforced a hard 7am–4pm ET session gate that matched its training distribution. v7 was trained on all-hours data, and the session context is encoded via the `in_optimal_session` CISD feature and `sess_id` sequence inputs. Adding a session gate to v7 reduces PT hits (tested: 3/6 with gate vs 4/6 without) and is not recommended.

### Gap-Through-Stop Risk

The bot sizes to the stop level, but macro events (CPI, FOMC, NFP) can gap through the stop. Actual loss in these cases may be 3–6× the intended risk budget. Avoid trading on known macro calendar days.

---

## License & Warranty

Provided "as is" without warranty. The author is not responsible for financial losses.

**USE AT YOUR OWN RISK.**
