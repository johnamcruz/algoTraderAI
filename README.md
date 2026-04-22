# AI Algo Trading Bot

![Python Tests](https://github.com/johnamcruz/algoTraderAI/actions/workflows/tests.yml/badge.svg)

## Overview

A Python algorithmic trading bot for futures markets (MNQ, MES, MGC) via the TopstepX/ProjectX platform. Uses a **CISD+OTE** (Change in State of Delivery + Optimal Trade Entry) AI strategy for entry signals, with dynamic risk-based position sizing.

**Active model:** `cisd_ote_hybrid_v5_1.onnx`

### Key Features

- 🧠 **CISD+OTE Strategy v5.1** — Zone-based AI entries using ICT concepts (liquidity sweeps, displacement, OTE retracements). Trained on ES, NQ, RTY, YM, GC across London+NY session (7am–4pm ET).
- 📊 **Real-time Data Processing** — Live tick aggregation and bar generation (1, 3, or 5 minute bars)
- 💰 **Risk-Based Position Sizing** — Size contracts dynamically from a dollar risk budget; skip signals that exceed it
- 🔬 **Backtesting Mode** — Replay historical CSV data with realistic gap-open fills and wick-based exit simulation
- 🚦 **OTE Depth Gate** — Filters shallow zone touches; only enters when price has penetrated deep enough into the OTE zone (`entry_distance_pct >= 3.0` default)
- 📈 **Volatility Regime Gate** — Skips entries when market volatility is abnormally low (`vty_regime >= 0.75` default)
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
algoTrader.py              # Entry point — arg parsing, mode dispatch
├── trading_bot.py         # Live bot: SignalR, tick aggregation, order execution
├── simulation_bot.py      # Backtest bot: CSV replay, P&L tracking, signal analysis
├── trading_bot_base.py    # Shared logic: AI prediction loop, sizing, exits
├── strategy_cisd_ote.py   # CISD+OTE strategy: features, model, entry/filter logic
├── strategy_base.py       # Abstract base class for strategies
├── strategy_factory.py    # Strategy registry
├── bot_utils.py           # Auth, logging, tick value/size lookups
├── config_loader.py       # YAML config loading and validation
└── backtest.py            # Scenario backtest runner (parallel, multi-symbol)
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
pip install onnxruntime pandas pandas-ta signalrcore requests numpy scikit-learn python-dateutil pyyaml
```

### Prepare Model Files

Place your trained model in the `models/` folder:

```
models/
└── cisd_ote_hybrid_v5_1.onnx     # active model
```

---

## Usage

### Backtest Runner (Recommended)

`backtest.py` runs predefined market-regime scenarios against historical data:

```bash
# Run all scenarios (MNQ default)
python3 backtest.py --parallel

# Run a single scenario
python3 backtest.py --scenario banking_2023

# Run on a different symbol
python3 backtest.py --symbol MGC --parallel

# Override entry confidence
python3 backtest.py --entry_conf 0.80 --parallel

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

### Direct Backtesting

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

### Live Trading

```bash
python algoTrader.py \
    --account TS001234SIM \
    --contract CON.F.US.MNQ.M26 \
    --username YourUsername \
    --apikey YourApiKey \
    --timeframe 5 \
    --strategy cisd-ote \
    --model models/cisd_ote_hybrid_v5_1.onnx \
    --entry_conf 0.70 \
    --risk_amount 300
```

### YAML Config (Recommended for Live)

```yaml
# configs/cisd_ote_mnq.yaml
username: "YourUsername"
apikey: "YourApiKey"
account: "TS001234SIM"
contract: "CON.F.US.MNQ.M26"
timeframe: 5

strategy: "cisd-ote"
model: "models/cisd_ote_hybrid_v5_1.onnx"

entry_conf: 0.70
risk_amount: 300
high_conf_multiplier: 2.0   # extend profit target to 4R at ≥90% confidence
```

```bash
python algoTrader.py --config configs/cisd_ote_mnq.yaml
```

---

## CLI Arguments

### Core

| Argument | Description | Default |
|----------|-------------|---------|
| `--strategy` | Strategy name (`cisd-ote`) | `supertrend` |
| `--model` | Path to ONNX model | — |
| `--contract` | Contract ID (e.g. `CON.F.US.MNQ.M26`) | — |
| `--timeframe` | Bar timeframe in minutes (1, 3, 5) | `5` |
| `--entry_conf` | Min AI confidence to enter (0.0–1.0) | `0.9` |

### Risk & Sizing

| Argument | Description | Default |
|----------|-------------|---------|
| `--risk_amount` | Max dollars to risk per trade (dynamic sizing) | None |
| `--size` | Fixed contract count (used when `--risk_amount` not set) | `1` |
| `--max_contracts` | Cap on contracts per trade | `15` |
| `--high_conf_multiplier` | Extend profit target at ≥90% confidence (risk unchanged) | `1.0` |
| `--max_loss` | Stop bot when cumulative loss hits this (dollars) | `3000` |
| `--profit_target` | Stop bot when cumulative P&L reaches this (dollars) | `6000` |
| `--min_stop_atr` | Minimum stop size as ATR multiple (prevents micro stops) | `0.5` |
| `--min_stop_pts` | Minimum stop size in points (floor) | `1.0` |

### Entry Filters (CISD+OTE)

| Argument | Description | Default |
|----------|-------------|---------|
| `--min_vty_regime` | Regime gate: skip entries when `atr14/atr_ma50` is below this (0.0 = off) | `0.75` |
| `--min_entry_distance` | OTE depth gate: skip signals where price hasn't penetrated deep enough into zone (0.0 = off) | `3.0` |

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

## CISD+OTE Strategy v5.1

**Concept:** Detects ICT-style liquidity events and enters in the direction of institutional order flow.

1. **CISD** — Identifies a Change in State of Delivery: a prior high/low is swept (liquidity grab), followed by a strong displacement candle closing back through the level
2. **OTE Zone** — Marks the 61.8–78.6% Fibonacci retracement of the displacement leg as the optimal entry zone
3. **AI Filter** — An ONNX classifier (`cisd_ote_hybrid_v5_1.onnx`) confirms the setup using 28 CISD features + 256-dim FFM backbone embeddings
4. **Entry Gates** — Three layered filters before entry is allowed:
   - Session gate: 7am–4pm ET only (London open through RTH close — matches training distribution)
   - Volatility regime gate: `vty_regime >= 0.75` (blocks entries in abnormally quiet markets)
   - OTE depth gate: `entry_distance_pct >= 3.0` (filters shallow zone touches; winners average 3.9–4.5 vs losers 2.1–2.9)
5. **Stop/Target** — Zone boundaries define the stop; target is 2R by default, extended to 4R on ≥90% confidence signals

**Model training details (v5.1):**
- Trained on ES, NQ, RTY, YM, GC (full-size) + micro equivalents (10 instruments)
- Session window: 7am–4pm ET (London open + full RTH)
- `in_optimal_session` feature marks 9–11am ET as 1.0 (NY prime window encoded as feature, not hard gate)
- Walk-forward folds F1–F4; ONNX exported from F4 (train end: 2025-01-01)
- 2-class output: 0=noise, 1=signal (direction from `zone_is_bullish` feature)

**Zone lifecycle:**
- A zone is consumed when price enters it — it won't re-trigger
- A stop-loss clears all active zones (start fresh)
- Pending entries are cancelled if the bar opens more than one stop-distance away (gap invalidation)

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
- At ≥90% confidence, `--high_conf_multiplier` extends the profit target (e.g. `2.0` → 4R); risk per trade is always `risk_amount`

### Gap Risk

Sizing targets the stop level. If a bar opens beyond the stop (gap), the exit fills at open — actual loss may exceed the risk budget. This is realistic market behavior.

---

## Per-Trade Signal Analysis

After each backtest, a winners-vs-losers feature table is printed:

```
Feature                   Winners     Losers      Delta
--------------------------------------------------------
signal_prob                0.7888     0.7566    +0.0322
entry_distance_pct         4.3413     1.7738    +2.5675
vty_regime                 1.2528     1.3659    -0.1131
...
```

This was used to identify `entry_distance_pct` as the most reliable separator between winning and losing trades, leading to the OTE depth gate (`--min_entry_distance 3.0`).

---

## Monitoring

```bash
# Tail logs
tail -f logs/bot_MNQ_YYYYMMDD.log

# Filter entries/exits
grep "ENTRY\|EXIT" logs/bot_MNQ_YYYYMMDD.log

# Check errors
grep "ERROR" logs/bot_MNQ_YYYYMMDD.log
```

---

## Known Issues & Limitations

### Session Gate Is a Hard Boundary

The model was trained exclusively on 7am–4pm ET data. Out-of-session signals will produce unreliable confidence scores — the model has never seen overnight bars as entry candidates. Do not disable the session gate.

### Gap-Through-Stop Risk

The bot sizes to the stop level, but macro events (CPI, FOMC, NFP) can gap through the stop. Actual loss in these cases may be 3–6× the intended risk budget. Avoid trading on known macro calendar days.

### 2021 Out-of-Sample — Low-Vol Grind

2021 (OOS control year) shows elevated trade counts and losing outcomes with the gate at `min_entry_distance=3.0` (30 trades, ~27% win rate). This reflects the model's weakness in low-volatility structural uptrends where displacement moves lack follow-through. The `vty_regime` gate reduces damage but doesn't eliminate it.

### Bear Markets — Model Over-Fires on Bounces

In persistent downtrends (2022), the model fires bullish CISD setups on every pullback bounce. The OTE depth gate (`min_entry_distance=3.0`) significantly improves this (167 → 53 trades, 37% → 47% win rate, -$313 → +$3,568) but the bear regime is still the weakest period.

---

## License & Warranty

Provided "as is" without warranty. The author is not responsible for financial losses.

**USE AT YOUR OWN RISK.**
