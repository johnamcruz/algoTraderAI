# AI Algo Trading Bot

![Python Tests](https://github.com/johnamcruz/algoTraderAI/actions/workflows/tests.yml/badge.svg)

## Overview

A Python algorithmic trading bot for futures markets (MNQ, MES, MGC) via the TopstepX/ProjectX platform. Supports multiple AI strategies: the default **CISD+OTE v7** (ICT-style zone entries), **SuperTrend v1** (trend-following with HTF alignment), and **VWAP Reversion v1** (mean-reversion at statistical extremes).

**Default model:** `cisd_ote_hybrid_v7.onnx` — FFM Hybrid Transformer

### Key Features

- 🧠 **CISD+OTE Strategy v7** — ICT-style zone detection with a 96-bar FFM Transformer backbone. Risk head predicts per-trade R:R; TP set to `int(predicted_rr) × R`.
- 📈 **SuperTrend Strategy v1** — 5m SuperTrend(10, 2.0) flip + 1h HTF alignment. Same FFM backbone and risk head — TP set to `int(predicted_rr) × R`. Trained on NQ, ES, GC, RTY, YM.
- 🔁 **VWAP Reversion Strategy v1** — Mean-reversion entries at statistical VWAP extremes. Same FFM backbone and risk head. High win rate, low trade frequency.
- 📊 **Real-time Data Processing** — Live tick aggregation and bar generation (1, 3, or 5 minute bars)
- 💰 **Risk-Based Position Sizing** — Size contracts dynamically from a dollar risk budget; skip signals that exceed it
- 🔬 **Backtesting Mode** — Replay historical CSV data with realistic gap-open fills and wick-based exit simulation
- 🎯 **RR Gate** — Filters low-expectancy signals; only enters when the model predicts `>= min_risk_rr` (default 2.0)
- 📋 **Per-Trade Signal Analysis** — Logs feature averages for winners vs losers after each backtest run
- ⚙️ **YAML Configuration** — All parameters configurable via YAML file; CLI args override
- 🔄 **Backtest Runner** — `backtest.py` runs predefined market-regime scenarios (bear, recovery, banking crisis, selloff, OOS) in parallel or sequentially

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
│   ├── strategy_st_trend_v1.py  # SuperTrend v1: FFM Transformer backbone + risk head TP
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

Place your trained models in the `models/` folder:

```
models/
├── cisd_ote_hybrid_v7.onnx       # CISD+OTE v7 — default (recommended)
├── st_trend_v1.onnx              # SuperTrend v1 — trend-following
├── vwap_v1.onnx                  # VWAP Reversion v1 — mean-reversion
└── cisd_ote_hybrid_v5_1.onnx     # CISD+OTE v5.1 — legacy
```

---

## Usage

### Backtest Runner (Recommended)

`backtest.py` runs predefined market-regime scenarios against historical data:

```bash
# Run all scenarios — CISD+OTE v7 (default)
python3 backtest.py --parallel

# Run all scenarios — SuperTrend v1
python3 backtest.py --strategy supertrend --parallel

# Run a single scenario
python3 backtest.py --scenario banking_2023

# Run on a different symbol
python3 backtest.py --symbol MES --parallel

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
| `recent_30d` | Rolling 30-day | Most recent 30-day window |
| `recent_60d` | Rolling 60-day | Most recent 60-day window |
| `recent_90d` | Rolling 90-day | Most recent 90-day window |
| `recent_120d` | Rolling 120-day | Most recent 120-day window |
| `recent_180d` | Rolling 180-day | Most recent 180-day window |

**Supported symbols:** `MNQ`, `MES`, `MGC`

### Direct Backtesting — CISD+OTE v7 (Default)

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

### Direct Backtesting — SuperTrend v1

```bash
python algoTrader.py \
    --backtest \
    --backtest_data data/NQ_continuous_5min.csv \
    --contract CON.F.US.MNQ.M26 \
    --strategy supertrend \
    --model models/st_trend_v1.onnx \
    --entry_conf 0.80 \
    --min_risk_rr 2.0 \
    --risk_amount 200
```

### Direct Backtesting — VWAP Reversion v1

```bash
python algoTrader.py \
    --backtest \
    --backtest_data data/NQ_continuous_5min.csv \
    --contract CON.F.US.MNQ.M26 \
    --strategy vwap \
    --model models/vwap_v1.onnx \
    --entry_conf 0.70 \
    --min_risk_rr 4.0 \
    --risk_amount 200
```

### Direct Backtesting — CISD+OTE v5.1 (Legacy)

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

### Live Trading — CISD+OTE v7

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

### Live Trading — SuperTrend v1

```bash
python algoTrader.py \
    --account TS001234SIM \
    --contract CON.F.US.MNQ.M26 \
    --username YourUsername \
    --apikey YourApiKey \
    --timeframe 5 \
    --strategy supertrend \
    --model models/st_trend_v1.onnx \
    --entry_conf 0.80 \
    --min_risk_rr 2.0 \
    --risk_amount 200
```

### Live Trading — VWAP Reversion v1

```bash
python algoTrader.py \
    --account TS001234SIM \
    --contract CON.F.US.MES.M26 \
    --username YourUsername \
    --apikey YourApiKey \
    --timeframe 5 \
    --strategy vwap \
    --model models/vwap_v1.onnx \
    --entry_conf 0.70 \
    --min_risk_rr 4.0 \
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
| `--strategy` | `cisd-ote7` (default), `supertrend`, or `cisd-ote` (legacy) | — |
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
| `--profit_target` | Stop bot when cumulative P&L reaches this (dollars) | `12000` |
| `--min_stop_atr` | Minimum stop size as ATR multiple (prevents micro stops) | `0.5` |
| `--min_stop_pts` | Minimum stop size in points (floor) | `1.0` |

### Entry Filters

| Argument | Strategy | Description | Default |
|----------|----------|-------------|---------|
| `--min_risk_rr` | all | Skip signals when model's predicted R:R is below this | `2.0` |
| `--min_vty_regime` | `cisd-ote` | Regime gate: skip entries when `atr14/atr_ma50` is below this (0.0 = off) | `0.75` |
| `--min_entry_distance` | `cisd-ote` | OTE depth gate: minimum zone penetration depth (0.0 = off) | `3.0` |
| `--high_conf_multiplier` | `cisd-ote` only | Extend profit target at ≥90% confidence (disabled for v7 and supertrend) | `1.0` |

### Backtesting

| Argument | Description | Default |
|----------|-------------|---------|
| `--backtest` | Run in backtesting mode | — |
| `--backtest_data` | Path to OHLCV CSV file | — |
| `--start-date` | Start date (`YYYY-MM-DD`) | None |
| `--end-date` | End date (`YYYY-MM-DD`) | None |
| `--no-profit-target` | Disable session profit target cap — run the full date range without stopping early | — |
| `--simulation-days` | Limit backtest to the last N days of the CSV | None |

### Live Only

| Argument | Description |
|----------|-------------|
| `--account` | TopstepX account ID |
| `--username` | TopstepX username |
| `--apikey` | TopstepX API key |

---

## Strategy Comparison

Three strategies are available. CISD+OTE v7 is the default; SuperTrend v1 and VWAP Reversion v1 are alternatives suited to different market regimes.

| | `cisd-ote7` (default) | `supertrend` | `vwap` | `cisd-ote` (legacy) |
|---|---|---|---|---|
| **Model** | `cisd_ote_hybrid_v7.onnx` | `st_trend_v1.onnx` | `vwap_v1.onnx` | `cisd_ote_hybrid_v5_1.onnx` |
| **Signal** | CISD zone + OTE entry | ST(10,2.0) flip + 1h HTF alignment | VWAP statistical extreme reversion | CISD zone + OTE entry |
| **Backbone** | 96-bar FFM Transformer | 96-bar FFM Transformer | 96-bar FFM Transformer | 64-bar FFM Transformer |
| **Profit target** | `int(predicted_rr) × R` | `int(predicted_rr) × R` | `int(predicted_rr) × R` | Fixed 2R (4R at ≥90% conf) |
| **Entry confidence** | 0.80 recommended | 0.80 recommended | 0.70 recommended | 0.70–0.85 |
| **RR gate** | `--min_risk_rr 2.0` | `--min_risk_rr 2.0` | `--min_risk_rr 4.0` | Not applicable |
| **Trained on** | NQ, ES, GC, RTY, YM | NQ, ES, GC, RTY, YM | NQ, ES, GC, RTY, YM | NQ only |
| **Trades/month (MNQ)** | ~20–30 | ~10–15 | ~5–10 | Fewer |
| **Best for** | Default — zone reversals | Trend-following regimes | Range-bound / mean-reversion | Legacy comparison |

**CISD+OTE v7 is the recommended default.** SuperTrend v1 suits strong trending conditions. VWAP Reversion v1 is a high-selectivity strategy for mean-reversion — fewest signals, highest RR gate. All three use the same `int(predicted_rr) × R` TP logic with no artificial ceiling.

---

## Multi-Ticker Setup

All strategies are designed to run simultaneously across uncorrelated instruments. Mix strategy types across tickers for diversification across both signal type and underlying:

| Ticker | Contract | Strategy | Notes |
|--------|----------|----------|-------|
| MNQ | `CON.F.US.MNQ.M26` | `supertrend` or `cisd-ote7` | Primary — highest signal quality |
| MES | `CON.F.US.MES.M26` | `cisd-ote7` or `vwap` | Correlated to MNQ — adds trade frequency |
| MGC | `CON.F.US.MGC.M26` | `cisd-ote7` | Uncorrelated to equities — genuine diversification |

Run each ticker in a separate terminal or process.

---

## Risk Management

### Dynamic Sizing (Recommended)

Set `--risk_amount` and the bot sizes each trade to risk exactly that many dollars at the stop level:

```
contracts = floor(risk_amount / (stop_ticks × tick_value))
```

- If even 1 contract exceeds the budget, the signal is skipped
- `--min_stop_atr` and `--min_stop_pts` prevent unrealistically tight stops from inflating size
- For `cisd-ote7` and `supertrend`, profit target is set by the risk head — `high_conf_multiplier` is always disabled

### Gap Risk

Sizing targets the stop level. If a bar opens beyond the stop (gap), the exit fills at open — actual loss may exceed the risk budget. This is realistic market behavior.

---

## Per-Trade Signal Analysis

After each backtest, a winners-vs-losers feature table is printed:

```
Feature                   Winners     Losers      Delta
--------------------------------------------------------
confidence                 0.8698     0.9061    -0.0363
risk_rr                    5.9561     1.0173    +4.9388
signal_atr                20.7646    48.4203   -27.6558
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

### Persistent Trend Blindness (CISD v7)

v7's 96-bar context window covers ~8 hours. In sustained multi-week downtrends (e.g. the 2022 bear market), the model fires bullish CISD setups because local structure looks valid — it has no awareness that the daily or weekly trend is bearish. SuperTrend is less susceptible to this since HTF alignment is an explicit input feature.

### Session Gate Removed in v7

v5.1 enforced a hard 7am–4pm ET session gate that matched its training distribution. v7 was trained on all-hours data, and the session context is encoded via the `in_optimal_session` CISD feature and `sess_id` sequence inputs. Adding a session gate to v7 reduces PT hits and is not recommended.

### Gap-Through-Stop Risk

The bot sizes to the stop level, but macro events (CPI, FOMC, NFP) can gap through the stop. Actual loss in these cases may be 3–6× the intended risk budget. Avoid trading on known macro calendar days.

---

## License & Warranty

Provided "as is" without warranty. The author is not responsible for financial losses.

**USE AT YOUR OWN RISK.**
