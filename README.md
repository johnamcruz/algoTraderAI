# AI Algo Trading Bot

## Overview

This Python script implements a **real-time algorithmic trading bot** for futures markets (ES, NQ, YM, RTY via TopstepX platform) with a **pluggable strategy architecture**. You can easily switch between different AI trading strategies or create your own custom strategies without modifying the core bot code.

### Key Features

- 🔌 **Pluggable AI Strategies** - Switch strategies via command-line argument
- 🤖 **Multiple Strategies Included** - Squeeze V3, Pivot Reversal, and more
- 📊 **Real-time Data Processing** - Live tick aggregation and bar generation
- 🎯 **AI-Powered Predictions** - LSTM models for entry signals
- 💰 **Risk Management** - ATR-based stops and profit targets
- 🛠️ **Easy to Extend** - Create custom strategies using provided template

---

## ⚠️ Disclaimer

**This is a trading bot implementation. Trading futures involves substantial risk of loss and is not suitable for all investors.**

- Use this code **at your own risk**
- Past performance is **not indicative of future results**
- **Paper trading is highly recommended** before going live
- Only use risk capital for trading

---

## Architecture

The bot separates **trading infrastructure** from **strategy logic**:

```
Trading Bot (Core)
├── Connection Management
├── Tick Aggregation
├── Order Execution
└── Position Management

Strategy (Pluggable)
├── Feature Calculation
├── AI Model Inference
└── Entry Signal Logic
```

This means you can:
- ✅ Use different AI strategies without changing bot code
- ✅ Create custom strategies easily
- ✅ Test strategies independently
- ✅ Switch strategies via command-line

---

## Installation

### 1. Requirements

- **Python 3.10+** (managed via `pyenv` recommended)
- Virtual environment recommended

### 2. Clone/Download

Download these files to your project directory:

```
your_trading_folder/
├── algoTrader.py              # Main trading bot
├── strategy_base.py           # Base class for strategies
├── strategy_factory.py        # Strategy creation
├── strategy_squeeze_v3.py     # Squeeze V3 strategy
├── strategy_pivot_reversal.py # Pivot Reversal strategy
└── strategy_template.py       # Template for custom strategies
```

### 3. Create Virtual Environment

Navigate to your project directory:

```bash
cd /path/to/your_trading_folder
```

Create and activate virtual environment:

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
# macOS/Linux:
source venv/bin/activate

# Windows:
.\venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install onnxruntime pandas pandas-ta signalrcore requests numpy scikit-learn python-dateutil
```

### 5. Get API Credentials

Obtain your **TopstepX** credentials:
- Username
- API Key

⚠️ **Never hardcode credentials** - always use command-line arguments

### 6. Prepare Model Files

You need two files for each strategy:

1. **ONNX model file** (`.onnx`) - Trained AI model
2. **Scaler file** (`.pkl`) - Feature scaling parameters

Organize in a `models/` folder:

```
your_trading_folder/
├── models/
│   ├── squeeze_v3_model.onnx
│   ├── squeeze_v3_scaler.pkl
│   ├── pivot_reversal_model.onnx
│   └── pivot_reversal_scaler.pkl
└── ...
```

---

## Usage

### Basic Command Structure

```bash
python algoTrader.py \
    --account <ACCOUNT_ID> \
    --contract <CONTRACT_ID> \
    --size <SIZE> \
    --username <USERNAME> \
    --apikey <API_KEY> \
    --strategy <STRATEGY_NAME> \
    --model <MODEL_PATH> \
    --scaler <SCALER_PATH> \
    --entry_conf <CONFIDENCE> \
    --adx_thresh <ADX> \
    --stop_atr <STOP_MULT> \
    --target_atr <TARGET_MULT>
```

### Required Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--account` | TopstepX account ID | `TS001234SIM` |
| `--contract` | Full contract ID | `CON.F.US.RTY.Z25` |
| `--size` | Number of contracts | `1` |
| `--username` | TopstepX username | `YourUsername` |
| `--apikey` | TopstepX API key | `YourApiKey` |
| `--strategy` | Strategy name | `squeeze_v3` or `pivot_reversal` |
| `--model` | Path to ONNX model | `models/squeeze_v3_model.onnx` |
| `--scaler` | Path to scaler file | `models/squeeze_v3_scaler.pkl` |

### Optional Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--timeframe` | Bar timeframe (1, 3, or 5 min) | `5` |
| `--entry_conf` | Min confidence (0.0-1.0) | `0.60` |
| `--adx_thresh` | Min ADX for entry | `20` |
| `--stop_atr` | Stop loss (x ATR) | `1.5` |
| `--target_atr` | Profit target (x ATR) | `2.0` |
| `--enable_trailing_stop` | Use trailing stop | `False` |
| `--pivot_lookback` | Pivot lookback (pivot_reversal only) | `5` |
| `--market_hub` | ProjectX MarketHub URL | `https://rtc.topstepx.com/hubs/market` |
| `--base_url` | ProjectX Base URL | `https://api.topstepx.com/api` |

---

## Example Usage

### Example 1: Squeeze V3 Strategy (RTY)

```bash
python algoTrader.py \
    --account TS001234SIM \
    --contract CON.F.US.RTY.Z25 \
    --size 1 \
    --username YourUsername \
    --apikey YourApiKey \
    --timeframe 5 \
    --strategy squeeze_v3 \
    --model models/squeeze_v3_model.onnx \
    --scaler models/squeeze_v3_scaler.pkl \
    --entry_conf 0.55 \
    --adx_thresh 25 \
    --stop_atr 2.0 \
    --target_atr 3.0
```

### Example 2: Pivot Reversal Strategy (ES)

```bash
python algoTrader.py \
    --account TS001234SIM \
    --contract CON.F.US.ES.Z25 \
    --size 1 \
    --username YourUsername \
    --apikey YourApiKey \
    --timeframe 5 \
    --strategy pivot_reversal \
    --model models/pivot_reversal_model.onnx \
    --scaler models/pivot_reversal_scaler.pkl \
    --entry_conf 0.70 \
    --adx_thresh 20 \
    --stop_atr 1.0 \
    --target_atr 2.0 \
    --pivot_lookback 5
```

### Stopping the Bot

Press **Ctrl+C** to stop gracefully.

---

## Available Strategies

### Squeeze V3

**Focus:** Bollinger Band / Keltner Channel compression with momentum analysis

**Features (9):**
- Compression level
- Squeeze duration
- BB/ATR expanding
- Price position
- RSI, momentum, volume surge, body strength

**Best For:**
- Range breakouts
- Volatility expansion plays
- Trending moves after consolidation

**Recommended Settings:**
- Entry Confidence: 0.55-0.60
- ADX Threshold: 20-25
- Stop: 2.0 ATR
- Target: 3.0 ATR

### Pivot Reversal

**Focus:** Pivot point breaks and rejections with trend context

**Features (24):**
- Pivot distances and bars since pivot
- Break/rejection signals
- Candle characteristics
- Trend and momentum indicators

**Best For:**
- Support/resistance bounces
- Pivot break continuations
- Mean reversion at extremes

**Recommended Settings:**
- Entry Confidence: 0.65-0.75
- ADX Threshold: 15-20
- Stop: 1.0-1.5 ATR
- Target: 2.0-2.5 ATR

---

## Creating Custom Strategies

Want to create your own strategy? It's easy!

### Step 1: Copy Template

Copy `strategy_template.py` to `strategy_my_custom.py`

### Step 2: Define Features

```python
def get_feature_columns(self) -> List[str]:
    return [
        'my_feature_1',
        'my_feature_2',
        'my_feature_3'
    ]
```

### Step 3: Calculate Features

```python
def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
    df['my_feature_1'] = ...  # Your calculation
    df['my_feature_2'] = ...
    df['my_feature_3'] = ...
    return df
```

### Step 4: Implement Model Loading

```python
def load_model(self):
    self.model = onnxruntime.InferenceSession(self.model_path)

def load_scaler(self):
    with open(self.scaler_path, 'rb') as f:
        scalers = pickle.load(f)
    self.scaler = scalers[self.contract_symbol]
```

### Step 5: Define Entry Logic

```python
def should_enter_trade(self, prediction, confidence, bar, entry_conf, adx_thresh):
    if confidence < entry_conf:
        return False, None
    
    # Your custom filters here
    
    if prediction == 1:
        return True, 'LONG'
    elif prediction == 2:
        return True, 'SHORT'
    return False, None
```

### Step 6: Register Strategy

In `strategy_factory.py`:

```python
from strategy_my_custom import MyCustomStrategy

class StrategyFactory:
    STRATEGIES = {
        'squeeze_v3': SqueezeV3Strategy,
        'pivot_reversal': PivotReversalStrategy,
        'my_custom': MyCustomStrategy,  # Add this
    }
```

### Step 7: Use Your Strategy

```bash
python algoTrader.py \
    --strategy my_custom \
    --model models/my_model.onnx \
    --scaler models/my_scaler.pkl \
    ...
```

See `strategy_template.py` for a complete example with comments.

---

## Parameter Guidelines

### Entry Confidence

Controls how confident the AI must be before entering:

| Setting | Confidence | Trade Frequency | Risk |
|---------|-----------|-----------------|------|
| Conservative | 0.65-0.75 | Low | Lower |
| Balanced | 0.55-0.65 | Medium | Medium |
| Aggressive | 0.45-0.55 | High | Higher |

### ADX Threshold

Controls minimum trend strength:

| Setting | ADX | Use When |
|---------|-----|----------|
| Strong Trends | 25-30 | Trending markets |
| Moderate Trends | 20-25 | Mixed conditions |
| Any Trend | 15-20 | Range-bound markets |

### Stop/Target Multipliers

Must align with how your model was trained:

| Profile | Stop ATR | Target ATR | Risk/Reward |
|---------|----------|------------|-------------|
| Conservative | 2.0-2.5 | 2.0-2.5 | 1:1 |
| Balanced | 1.5-2.0 | 2.5-3.0 | 1:1.5-2 |
| Aggressive | 1.0-1.5 | 3.0-4.0 | 1:2-3 |

⚠️ **Critical:** `--target_atr` must match your model's training target!

---

## Troubleshooting

### Connection Issues

**Bot won't connect:**
- Verify username/API key
- Check internet connection
- Confirm TopstepX services are up

### File Issues

**"Model file not found":**
- Use absolute paths: `/full/path/to/model.onnx`
- Check file exists: `ls models/`
- Verify file permissions

**"Scaler not found":**
- Ensure scaler matches model
- Check contract symbol in scaler file
- Verify pickle file isn't corrupted

### Strategy Issues

**"Unknown strategy":**
- Check spelling: `squeeze_v3` not `squeeze-v3`
- Verify strategy registered in `strategy_factory.py`
- Run: `python algoTrader.py --help` to see available strategies

### Trading Issues

**No trades:**
- Lower `--entry_conf` (try 0.50)
- Lower `--adx_thresh` (try 15)
- Check market is active (not Asian hours)
- Verify strategy conditions occur in current market

**Too many losses:**
- Increase `--entry_conf` (try 0.65)
- Increase `--adx_thresh` (try 25)
- Paper trade different parameters
- Review if model/scaler are correct

---

## Performance Monitoring

### Key Metrics

Track these metrics while bot is running:

- **Win Rate**: Target 50-60%
- **Profit Factor**: Target > 1.5
- **Avg Win vs Avg Loss**: Win should be larger
- **Max Drawdown**: Monitor consecutive losses
- **Trade Frequency**: 5-15 setups/week expected

### Logging

The bot logs all activity. Monitor for:
- Entry signals and confidence levels
- Exit reasons (stop/target/time)
- Technical indicator values
- Position P&L

---

## Best Practices

### Before Going Live

1. ✅ **Paper trade** for 2-4 weeks minimum
2. ✅ **Compare results** to backtest expectations
3. ✅ **Test different parameters** on paper account
4. ✅ **Understand the strategy** you're using
5. ✅ **Have a plan** for drawdowns

### Risk Management

- 💰 Only trade with risk capital
- 📊 Start with 1 contract per trade
- 🎯 Set maximum daily loss limits
- ⏰ Avoid trading during news events
- 📉 Stop trading after 3 consecutive losses

### System Requirements

- 🔌 Stable internet connection
- 💻 Computer that stays on during trading hours
- 📱 Mobile alerts for order fills
- 🔄 Backup power supply recommended

---

## Files Included

| File | Description |
|------|-------------|
| `algoTrader.py` | Main trading bot |
| `strategy_base.py` | Abstract base class for strategies |
| `strategy_factory.py` | Creates strategy instances |
| `strategy_squeeze_v3.py` | Squeeze V3 strategy |
| `strategy_pivot_reversal.py` | Pivot Reversal strategy |
| `strategy_template.py` | Template for custom strategies |
| `README.md` | This file |

---

## Support & Resources

### Documentation

- See `strategy_template.py` for custom strategy examples
- Check docstrings in strategy files for details
- Review `strategy_base.py` for required methods

### Common Questions

**Q: Can I run multiple strategies at once?**
A: Yes, run separate instances with different contracts or strategies.

**Q: How do I update a strategy?**
A: Just modify the strategy file - no need to change the bot code.

**Q: Can I use my own model?**
A: Yes! Create a custom strategy, train your model, export to ONNX.

**Q: What if my broker isn't TopstepX?**
A: If your prop firm utilizes the ProjectX API framework, simply update the Market HUB URL and the Base URL within your configuration files to match your broker's specific API endpoints.

---

## Version History

- **v1.0**: Initial release with hardcoded strategy
- **v2.0**: Refactored with pluggable strategy architecture
  - Added strategy base class
  - Included Squeeze V3 and Pivot Reversal
  - Added strategy template
  - Separated trading logic from AI logic

---

## License & Warranty

This software is provided "as is" without warranty of any kind. The author is not responsible for any financial losses.

Trading futures involves substantial risk. Only use risk capital.

---

**Happy Trading! 🚀**

*The best trade is often the one you don't take. Always prioritize capital preservation.*