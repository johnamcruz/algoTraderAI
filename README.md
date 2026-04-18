# AI Algo Trading Bot

## Overview

A Python algorithmic trading bot for futures markets (NQ, ES, RTY, etc.) via the TopstepX/ProjectX platform. Uses a **CISD+OTE** (Change in State of Delivery + Optimal Trade Entry) AI strategy for entry signals, with dynamic risk-based position sizing.

### Key Features

- 🧠 **CISD+OTE Strategy** — Zone-based AI entries using ICT concepts (liquidity sweeps, displacement, OTE retracements)
- 📊 **Real-time Data Processing** — Live tick aggregation and bar generation (1, 3, or 5 minute bars)
- 💰 **Risk-Based Position Sizing** — Size contracts dynamically from a dollar risk budget; skip signals that exceed it
- 🔬 **Backtesting Mode** — Replay historical CSV data with realistic gap-open fills and wick-based exit simulation
- ⚙️ **YAML Configuration** — Manage live and backtest configs via file; CLI args override

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
├── simulation_bot.py      # Backtest bot: CSV replay, P&L tracking
├── trading_bot_base.py    # Shared logic: AI prediction loop, sizing, exits
├── strategy_cisd_ote.py   # CISD+OTE strategy: features, model, entry logic
├── strategy_base.py       # Abstract base class for strategies
├── strategy_factory.py    # Strategy registry
├── bot_utils.py           # Auth, logging, tick value/size lookups
└── config_loader.py       # YAML config loading and validation
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

Place your trained model and scaler in the `models/` folder:

```
models/
├── cisd_ote_hybrid_v5_1.onnx
└── cisd_ote_hybrid_v5_1_scaler.pkl   # (if applicable)
```

---

## Usage

### Backtesting

```bash
python algoTrader.py \
    --backtest \
    --backtest_data data/NQ_continuous_5min.csv \
    --contract CON.F.US.ENQ.Z25 \
    --strategy cisd-ote \
    --model models/cisd_ote_hybrid_v5_1.onnx \
    --entry_conf 0.70 \
    --risk_amount 300 \
    --simulation-days 10
```

### Live Trading

```bash
python algoTrader.py \
    --account TS001234SIM \
    --contract CON.F.US.ENQ.Z25 \
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
# configs/cisd_ote_nq.yaml
username: "YourUsername"
apikey: "YourApiKey"
account: "TS001234SIM"
contract: "CON.F.US.ENQ.Z25"
timeframe: 5

strategy: "cisd-ote"
model: "models/cisd_ote_hybrid_v5_1.onnx"

entry_conf: 0.70
adx_thresh: 0
risk_amount: 300
high_conf_multiplier: 2.0   # extend profit target to 4R at ≥90% confidence (risk unchanged)
```

```bash
python algoTrader.py --config configs/cisd_ote_nq.yaml
```

---

## CLI Arguments

### Core

| Argument | Description | Default |
|----------|-------------|---------|
| `--strategy` | Strategy name (`cisd-ote`) | `supertrend` |
| `--model` | Path to ONNX model | — |
| `--contract` | Contract ID (e.g. `CON.F.US.ENQ.Z25`) | — |
| `--timeframe` | Bar timeframe in minutes (1, 3, 5) | `3` |
| `--entry_conf` | Min AI confidence to enter (0.0–1.0) | `0.9` |
| `--adx_thresh` | Min ADX for entry (0 = disabled) | `0` |

### Risk & Sizing

| Argument | Description | Default |
|----------|-------------|---------|
| `--risk_amount` | Max dollars to risk per trade (dynamic sizing) | None |
| `--size` | Fixed contract count (used when `--risk_amount` not set) | `1` |
| `--high_conf_multiplier` | Extend profit target by this factor at ≥90% confidence (risk unchanged) | `1.0` |
| `--stop_pts` | Stop loss in points (optional; strategy provides its own) | None |
| `--target_pts` | Profit target in points (optional; strategy provides its own) | None |

### Backtesting

| Argument | Description | Default |
|----------|-------------|---------|
| `--backtest` | Run in backtesting mode | — |
| `--backtest_data` | Path to OHLCV CSV file | — |
| `--simulation-days` | Limit backtest to first N days of data | None |
| `--profit_target` | Stop sim when cumulative P&L reaches this (dollars) | `6000` |
| `--max_loss` | Stop sim when cumulative loss hits this (dollars) | `3000` |

### Live Only

| Argument | Description |
|----------|-------------|
| `--account` | TopstepX account ID |
| `--username` | TopstepX username |
| `--apikey` | TopstepX API key |
| `--market_hub` | MarketHub URL (default: TopstepX) |
| `--base_url` | Base API URL (default: TopstepX) |

---

## CISD+OTE Strategy

**Concept:** Detects ICT-style liquidity events and enters in the direction of institutional order flow.

1. **CISD** — Identifies a Change in State of Delivery: a prior high/low is swept (liquidity grab), followed by a strong displacement candle closing back through the level
2. **OTE Zone** — Marks the 62–79% Fibonacci retracement of the displacement leg as the optimal entry zone
3. **AI Filter** — An ONNX classifier confirms the setup using ~30 price-action and market-structure features
4. **Entry** — Triggers when price retraces into the OTE zone and AI confidence exceeds `entry_conf`
5. **Stop/Target** — Zone boundaries define the stop; target is 2R by default, extended to 4R on ≥90% confidence signals (via `high_conf_multiplier`)

**Zone lifecycle:**
- A zone is consumed (signal fired) when price enters it — it won't re-trigger
- A stop-loss clears all active zones (start fresh)
- Pending entries are cancelled if the bar opens more than one stop-distance away from the signal close (zone no longer valid after a large gap)

---

## Risk Management

### Dynamic Sizing (Recommended)

Set `--risk_amount` and the bot sizes each trade to risk exactly that many dollars at the stop level:

```
contracts = floor(risk_amount / (stop_ticks × tick_value))
```

- If even 1 contract exceeds the budget, the signal is skipped
- The CISD+OTE strategy provides its own stop (zone boundary) — no need to set `--stop_pts`
- At ≥90% confidence, `--high_conf_multiplier` extends the profit target (e.g. `2.0` doubles the target from 2R → 4R); risk per trade is always `risk_amount` regardless of confidence

### Fixed Sizing

Omit `--risk_amount` and pass `--size N` to always trade N contracts.

### Gap Risk

Sizing targets the stop level. If a bar opens beyond the stop (gap), the exit fills at the open price — the actual loss may exceed the risk budget. This is realistic market behavior.

---

## Backtesting Details

### CSV Format

```
time,open,high,low,close,volume
2025-10-01 09:30:00,20000.00,20015.50,19995.25,20010.00,12345
```

- `time` column can be Unix timestamps (seconds) or datetime strings
- `volume` is optional (defaults to 0 if missing)

### Simulation Logic

1. Bar N closes → strategy generates signal
2. Bar N+1 opens → pending entry fills at open (or is cancelled if gap > stop)
3. Same bar: check gap-open exit first, then intrabar wick (high/low) exit
4. No same-bar entry+exit (entry bar is protected)

### Tick Values

Tick sizes and values are auto-detected from the contract ID — no `--tick_size` needed:

| Symbol | Tick Size | Tick Value | Point Value |
|--------|-----------|------------|-------------|
| ENQ (NQ) | 0.25 | $5.00 | $20/pt |
| EP (ES) | 0.25 | $12.50 | $50/pt |
| MNQ | 0.25 | $0.50 | $2/pt |
| MES | 0.25 | $1.25 | $5/pt |

---

## Monitoring

```bash
# Tail logs
tail -f bot_log.log

# Filter entries/exits
grep "ENTRY\|EXIT" bot_log.log

# Check errors
grep "ERROR" bot_log.log
```

Enable verbose output with `--debug`.

---

## Troubleshooting

**No trades executing**
- Lower `--entry_conf` (try 0.65–0.70)
- Ensure sufficient history in the CSV (200+ bars before first signal)
- Check logs for "warming up" messages

**Scaler/model errors**
- Verify the `.onnx` file path is correct
- Check logs for feature validation failures (`❌ Feature validation failed`)

**Connection issues (live)**
- Verify API credentials and account ID
- Confirm TopstepX services are operational

---

## License & Warranty

Provided "as is" without warranty. The author is not responsible for financial losses.

**USE AT YOUR OWN RISK.**
