# AI Algo Trading Bot

## Overview

This Python application implements a **real-time algorithmic trading bot** for futures markets (ES, NQ, YM, RTY via TopstepX/ProjectX platform) with a **pluggable strategy architecture**. Switch between different AI trading strategies or create custom strategies without modifying the core bot code.

### Key Features

- 🔌 **Pluggable AI Strategies** - Switch strategies via command-line or YAML config
- 🤖 **Six Production Strategies Included** - Squeeze V3, Pivot Reversal (3min/5min), VWAP Mean Reversion, Trend Pullback (V1/V2)
- 📊 **Real-time Data Processing** - Live tick aggregation and bar generation (1, 3, or 5 minute bars)
- 🎯 **AI-Powered Predictions** - LSTM/Transformer models for entry signals
- 💰 **Robust Risk Management** - ATR-based stops and profit targets with optional trailing stops
- 🛠️ **Easy to Extend** - Create custom strategies using the BaseStrategy template
- ⚙️ **YAML Configuration Support** - Manage multiple configurations easily

---

## ⚠️ Disclaimer

**This is a trading bot implementation. Trading futures involves substantial risk of loss and is not suitable for all investors.**

- Use this code **at your own risk**
- Past performance is **not indicative of future results**
- **Paper trading is highly recommended** before going live
- Only use risk capital you can afford to lose
- Understand the strategy you're deploying before going live

---

## Architecture

The bot separates **trading infrastructure** from **strategy logic**:

```
Trading Bot Core (trading_bot.py)
├── SignalR Connection Management
├── Real-time Tick Aggregation
├── OHLCV Bar Generation
├── Order Execution & Bracket Orders
└── Position & Risk Management

Strategy Layer (Pluggable)
├── Feature Calculation (Technical Indicators)
├── AI Model Inference (ONNX)
├── Entry Signal Logic
└── Strategy-Specific Filters

Configuration Layer
├── YAML Config Files
├── Command-Line Arguments
└── Parameter Validation
```

This architecture means you can:
- ✅ Use different AI strategies without changing bot code
- ✅ Create and test custom strategies independently
- ✅ Switch strategies via command-line or config file
- ✅ Run multiple bots with different configs simultaneously

---

## Installation

### 1. Requirements

- **Python 3.10+** (managed via `pyenv` recommended)
- Virtual environment recommended
- TopstepX or ProjectX-compatible API access

### 2. Project Structure

```
your_trading_folder/
├── algoTrader.py                      # Main entry point
├── trading_bot.py                     # Core bot engine
├── config_loader.py                   # YAML config loader
├── bot_utils.py                       # Utility functions
├── strategy_base.py                   # Abstract base class
├── strategy_factory.py                # Strategy registry
├── strategy_squeeze.py                # Squeeze V3 strategy
├── strategy_pivot_reversal_3min.py    # 3-min pivot reversal
├── strategy_pivot_reversal_5min.py    # 5-min pivot reversal
├── strategy_vwap_3min.py              # VWAP mean reversion
├── strategy_trend_pullback.py         # Trend pullback V1
├── strategy_trend_pullback2.py        # Trend pullback V2
├── configs/                           # YAML config files
│   ├── squeeze_v3_config.yaml
│   ├── pivot_reversal_3min_config.yaml
│   └── ...
└── models/                            # ONNX models & scalers
    ├── squeeze_v3_model.onnx
    ├── squeeze_v3_scaler.pkl
    └── ...
```

### 3. Create Virtual Environment

```bash
cd /path/to/your_trading_folder

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
pip install onnxruntime pandas pandas-ta signalrcore requests numpy scikit-learn python-dateutil pyyaml
```

### 5. Get API Credentials

Obtain your **TopstepX** or **ProjectX-compatible** credentials:
- Username
- API Key
- Account ID
- Market Hub URL (default: `https://rtc.topstepx.com/hubs/market`)
- Base API URL (default: `https://api.topstepx.com/api`)

⚠️ **Never hardcode credentials** - always use command-line arguments or config files

### 6. Prepare Model Files

Each strategy requires two files:

1. **ONNX model file** (`.onnx`) - Trained AI model
2. **Scaler file** (`.pkl`) - Feature normalization parameters

Organize in a `models/` folder:

```
models/
├── squeeze_v3_model.onnx
├── squeeze_v3_scaler.pkl
├── model_3min_pivot_reversal_v2_final.onnx
├── scalers_3min_pivot_reversal_v2_final.pkl
├── vwap_3min_model.onnx
├── vwap_3min_scaler.pkl
└── ...
```

---

## Usage

### Method 1: YAML Configuration (Recommended)

Create a YAML config file (e.g., `configs/my_strategy.yaml`):

```yaml
# Account & Authentication
username: "YourUsername"
apikey: "YourApiKey"
account: "TS001234SIM"
contract: "CON.F.US.NQ.Z25"
size: 1

# Connection URLs (optional - defaults shown)
market_hub: "https://rtc.topstepx.com/hubs/market"
base_url: "https://api.topstepx.com/api"

# Timeframe
timeframe: 3  # 1, 3, or 5 minutes

# Strategy Selection
strategy: "3min_pivot_reversal"
model: "models/model_3min_pivot_reversal_v2_final.onnx"
scaler: "models/scalers_3min_pivot_reversal_v2_final.pkl"

# Trading Parameters
entry_conf: 0.60      # Minimum AI confidence (0.0-1.0)
adx_thresh: 20        # Minimum ADX for entry
stop_atr: 1.5         # Stop loss multiplier (x ATR)
target_atr: 2.0       # Profit target multiplier (x ATR)
enable_trailing_stop: false

# Strategy-specific parameters (if needed)
pivot_lookback: 8     # For pivot_reversal strategies
```

Run with config file:

```bash
python algoTrader.py --config configs/my_strategy.yaml
```

Command-line arguments override config file values:

```bash
python algoTrader.py --config configs/my_strategy.yaml --entry_conf 0.65 --size 2
```

### Method 2: Command-Line Arguments

```bash
python algoTrader.py \
    --account TS001234SIM \
    --contract CON.F.US.NQ.Z25 \
    --size 1 \
    --username YourUsername \
    --apikey YourApiKey \
    --timeframe 3 \
    --strategy 3min_pivot_reversal \
    --model models/model_3min_pivot_reversal_v2_final.onnx \
    --scaler models/scalers_3min_pivot_reversal_v2_final.pkl \
    --entry_conf 0.60 \
    --adx_thresh 20 \
    --stop_atr 1.5 \
    --target_atr 2.0
```

### Required Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--account` | TopstepX account ID | `TS001234SIM` |
| `--contract` | Full contract ID | `CON.F.US.NQ.Z25` |
| `--size` | Number of contracts | `1` |
| `--username` | TopstepX username | `YourUsername` |
| `--apikey` | TopstepX API key | `YourApiKey` |
| `--strategy` | Strategy name | See Available Strategies |
| `--model` | Path to ONNX model | `models/model.onnx` |
| `--scaler` | Path to scaler file | `models/scaler.pkl` |

### Optional Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--config` | Path to YAML config file | None |
| `--timeframe` | Bar timeframe (1, 3, or 5 min) | `3` |
| `--entry_conf` | Min confidence (0.0-1.0) | `0.60` |
| `--adx_thresh` | Min ADX for entry | `20` |
| `--stop_atr` | Stop loss (x ATR) | `1.5` |
| `--target_atr` | Profit target (x ATR) | `2.0` |
| `--enable_trailing_stop` | Use trailing stop | `False` |
| `--pivot_lookback` | Pivot lookback period | `8` |
| `--market_hub` | ProjectX MarketHub URL | TopstepX default |
| `--base_url` | ProjectX Base API URL | TopstepX default |

---

## Available Strategies

The bot includes six production-ready strategies. View all available strategies:

```bash
python algoTrader.py --help
```

### 1. Squeeze V3 (`squeeze_v3`)

**Focus:** Bollinger Band / Keltner Channel compression with momentum analysis

**Features:** 9 features including compression level, squeeze duration, BB/ATR expansion, price position, RSI, momentum, volume surge, and body strength.

**Best For:**
- Range breakouts
- Volatility expansion plays
- Trending moves after consolidation

**Recommended Settings:**
```yaml
strategy: squeeze_v3
entry_conf: 0.55-0.60
adx_thresh: 20-25
stop_atr: 2.0
target_atr: 3.0
timeframe: 5
```

### 2. Pivot Reversal - 3 Minute (`3min_pivot_reversal`)

**Focus:** Pivot point breaks and rejections with trend context optimized for 3-minute charts

**Features:** 24 features including pivot distances, bars since pivot, break/rejection signals, candle characteristics, trend and momentum indicators.

**Best For:**
- Support/resistance bounces
- Pivot break continuations
- Mean reversion at extremes
- Scalping and quick reversals

**Recommended Settings:**
```yaml
strategy: 3min_pivot_reversal
model: models/model_3min_pivot_reversal_v2_final.onnx
scaler: models/scalers_3min_pivot_reversal_v2_final.pkl
entry_conf: 0.60-0.70
adx_thresh: 20
stop_atr: 1.5
target_atr: 2.0
pivot_lookback: 8
timeframe: 3
```

### 3. Pivot Reversal - 5 Minute (`5min_pivot_reversal`)

**Focus:** Pivot point strategy optimized for 5-minute charts with tighter filters

**Features:** Same 24 features as 3min version, calibrated for longer timeframe.

**Best For:**
- Swing trades
- Support/resistance bounces with confirmation
- Lower frequency, higher quality setups

**Recommended Settings:**
```yaml
strategy: 5min_pivot_reversal
entry_conf: 0.65-0.75
adx_thresh: 15-20
stop_atr: 1.0-1.5
target_atr: 2.0-2.5
pivot_lookback: 5
timeframe: 5
```

### 4. VWAP Mean Reversion (`vwap`)

**Focus:** Price deviations from daily VWAP with momentum and trend filters

**Features:** 15 features including momentum context (RSI, Stochastic, MACD), dynamic levels (EMAs, VWAP), trend/volatility (ADX, uptrend/downtrend), and mean reversion signals (overbought/oversold).

**Internal Filters:**
- ADX Range: 20-40 (avoids both chop and strong trends)
- Trend Alignment: Optional EMA50 filter
- VWAP Stretch: 0.5 ATR bands

**Best For:**
- Range-bound markets
- Mean reversion opportunities
- VWAP deviation trades
- Counter-trend entries

**Recommended Settings:**
```yaml
strategy: vwap
entry_conf: 0.55-0.65
adx_thresh: 20
stop_atr: 1.5-2.0
target_atr: 2.0-2.5
timeframe: 3
```

**Note:** This strategy has hardcoded internal filters that work best in ranging conditions. The ADX threshold parameter acts as an additional filter on top of the internal 20-40 ADX range.

### 5. Trend Pullback V1 (`trend_pullback`)

**Focus:** Trend continuations using EMA 15/40 pullback system with Transformer model

**Features:** 11 features including price position context, momentum/volatility (ADX, RSI, MACD), volume confirmation (CMF), and trend duration (the "genesis" feature).

**Entry Logic:** Requires BOTH model signal AND pullback trigger:
- Long: Close near EMA15 AND EMA15 > EMA40
- Short: Close near EMA15 AND EMA15 < EMA40

**Best For:**
- Trending markets
- Pullback entries in established trends
- EMA-based trend following
- Momentum continuation

**Recommended Settings:**
```yaml
strategy: trend_pullback
entry_conf: 0.60-0.70
adx_thresh: 20-25
stop_atr: 1.5-2.0
target_atr: 2.5-3.0
timeframe: 3 or 5
```

**Key Concept:** The strategy waits for price to pull back to the fast EMA (15) within a 0.5 ATR range while the trend is intact (fast > slow EMA). This provides low-risk entries in the direction of the trend.

### 6. Trend Pullback V2 (`trend_pullback2`)

**Focus:** Enhanced version of Trend Pullback V1 with additional features or refinements

**Best For:** Similar to V1, potentially with different model weights or feature engineering.

**Recommended Settings:**
```yaml
strategy: trend_pullback2
entry_conf: 0.60-0.70
adx_thresh: 20-25
stop_atr: 1.5-2.0
target_atr: 2.5-3.0
timeframe: 3 or 5
```

---

## Strategy Comparison Matrix

| Strategy | Timeframe | Features | Market Type | Win Rate Target | Risk/Reward |
|----------|-----------|----------|-------------|-----------------|-------------|
| Squeeze V3 | 5min | 9 | Breakout | 50-55% | 1:1.5 |
| 3min Pivot | 3min | 24 | Reversal/Bounce | 55-60% | 1:1.3 |
| 5min Pivot | 5min | 24 | Reversal/Swing | 55-60% | 1:1.5-2 |
| VWAP | 3min | 15 | Mean Reversion | 50-60% | 1:1.3-1.5 |
| Trend Pullback V1 | 3-5min | 11 | Trend Continuation | 55-65% | 1:1.5-2 |
| Trend Pullback V2 | 3-5min | 11 | Trend Continuation | 55-65% | 1:1.5-2 |

---

## Example Configurations

### Example 1: Conservative VWAP Trading (NQ)

```yaml
# configs/vwap_conservative_nq.yaml
username: "YourUsername"
apikey: "YourApiKey"
account: "TS001234SIM"
contract: "CON.F.US.NQ.Z25"
size: 1
timeframe: 3

strategy: "vwap"
model: "models/vwap_3min_model.onnx"
scaler: "models/vwap_3min_scaler.pkl"

entry_conf: 0.65    # Higher confidence = fewer trades
adx_thresh: 25      # Stronger trend requirement
stop_atr: 2.0       # Wider stop
target_atr: 2.5     # Conservative target
enable_trailing_stop: false
```

Run:
```bash
python algoTrader.py --config configs/vwap_conservative_nq.yaml
```

### Example 2: Aggressive 3min Pivot (ES)

```yaml
# configs/pivot_aggressive_es.yaml
username: "YourUsername"
apikey: "YourApiKey"
account: "TS001234SIM"
contract: "CON.F.US.ES.Z25"
size: 2
timeframe: 3

strategy: "3min_pivot_reversal"
model: "models/model_3min_pivot_reversal_v2_final.onnx"
scaler: "models/scalers_3min_pivot_reversal_v2_final.pkl"

entry_conf: 0.55    # Lower confidence = more trades
adx_thresh: 15      # Lower ADX = trade in any condition
stop_atr: 1.5       # Tight stop
target_atr: 2.0     # Standard target
pivot_lookback: 8
enable_trailing_stop: true  # Lock in profits
```

### Example 3: Trend Following (RTY)

```yaml
# configs/trend_pullback_rty.yaml
username: "YourUsername"
apikey: "YourApiKey"
account: "TS001234SIM"
contract: "CON.F.US.RTY.Z25"
size: 1
timeframe: 5

strategy: "trend_pullback"
model: "models/trend_pullback_v1_model.onnx"
scaler: "models/trend_pullback_v1_scaler.pkl"

entry_conf: 0.65
adx_thresh: 25      # Strong trend required
stop_atr: 2.0
target_atr: 3.0     # Let winners run
enable_trailing_stop: false
```

---

## Creating Custom Strategies

Want to create your own strategy? Follow these steps:

### Step 1: Create Strategy File

Create `strategy_my_custom.py` based on `strategy_base.py`:

```python
#!/usr/bin/env python3
from strategy_base import BaseStrategy
import pandas as pd
import numpy as np
import onnxruntime
import pickle
import logging
from typing import List, Tuple, Optional, Dict

class MyCustomStrategy(BaseStrategy):
    """Your custom trading strategy."""
    
    def __init__(self, model_path: str, scaler_path: str, contract_symbol: str):
        super().__init__(model_path, scaler_path, contract_symbol)
        logging.info("Initialized MyCustomStrategy")
    
    def get_feature_columns(self) -> List[str]:
        """Define your features."""
        return [
            'rsi',
            'macd_hist',
            'atr',
            'volume_ratio',
            # ... more features
        ]
    
    def get_sequence_length(self) -> int:
        """How many bars your model needs."""
        return 40  # or whatever your model was trained on
    
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators."""
        # Your feature calculations
        df['rsi'] = ta.rsi(df['close'], length=14)
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        # ... more calculations
        
        # Clean NaN values
        df = df.fillna(method='ffill').fillna(0)
        return df
    
    def load_model(self):
        """Load your ONNX model."""
        self.model = onnxruntime.InferenceSession(self.model_path)
        logging.info(f"Loaded model: {self.model_path}")
    
    def load_scaler(self):
        """Load your feature scaler."""
        with open(self.scaler_path, 'rb') as f:
            scalers = pickle.load(f)
        self.scaler = scalers[self.contract_symbol]
        logging.info(f"Loaded scaler for {self.contract_symbol}")
    
    def predict(self, df: pd.DataFrame) -> Tuple[int, float]:
        """Run model inference."""
        features = self.preprocess_features(df)
        seq_len = self.get_sequence_length()
        
        if len(features) < seq_len:
            return 0, 0.0  # Not enough data
        
        X = features[-seq_len:].reshape(1, seq_len, -1).astype(np.float32)
        
        input_name = self.model.get_inputs()[0].name
        output_name = self.model.get_outputs()[0].name
        logits = self.model.run([output_name], {input_name: X})[0]
        
        probs = self._softmax(logits[0])
        prediction = int(np.argmax(probs))
        confidence = float(probs[prediction])
        
        return prediction, confidence
    
    def should_enter_trade(
        self,
        prediction: int,
        confidence: float,
        bar: Dict,
        entry_conf: float,
        adx_thresh: float
    ) -> Tuple[bool, Optional[str]]:
        """Entry logic with your custom filters."""
        
        # Check confidence
        if confidence < entry_conf:
            return False, None
        
        # Check ADX (optional)
        adx = bar.get('adx', 0)
        if adx < adx_thresh:
            return False, None
        
        # Your custom filters here
        # ...
        
        # Map prediction to direction
        if prediction == 1:
            return True, 'LONG'
        elif prediction == 2:
            return True, 'SHORT'
        
        return False, None
    
    @staticmethod
    def _softmax(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
```

### Step 2: Register Strategy

In `strategy_factory.py`, add your strategy:

```python
from strategy_my_custom import MyCustomStrategy

class StrategyFactory:
    STRATEGIES = {
        'squeeze_v3': SqueezeV3Strategy,
        '3min_pivot_reversal': PivotReversal3minStrategy,
        '5min_pivot_reversal': PivotReversal5minStrategy,
        'vwap': VWAP3minStrategy,
        'trend_pullback': TrendPullbackStrategy,
        'trend_pullback2': TrendPullbackStrategy2,
        'my_custom': MyCustomStrategy,  # Add this line
    }
```

### Step 3: Create Config

Create `configs/my_custom.yaml`:

```yaml
username: "YourUsername"
apikey: "YourApiKey"
account: "TS001234SIM"
contract: "CON.F.US.NQ.Z25"
size: 1
timeframe: 3

strategy: "my_custom"
model: "models/my_custom_model.onnx"
scaler: "models/my_custom_scaler.pkl"

entry_conf: 0.60
adx_thresh: 20
stop_atr: 1.5
target_atr: 2.0
```

### Step 4: Test

```bash
python algoTrader.py --config configs/my_custom.yaml
```

---

## Parameter Tuning Guidelines

### Entry Confidence Levels

Controls how confident the AI must be before entering:

| Profile | Confidence | Trade Frequency | Precision | Recall |
|---------|-----------|-----------------|-----------|---------|
| Conservative | 0.70-0.80 | Very Low | High | Low |
| Balanced | 0.60-0.70 | Medium | Medium | Medium |
| Aggressive | 0.50-0.60 | High | Lower | High |
| Very Aggressive | 0.40-0.50 | Very High | Low | Very High |

**Recommendation:** Start with 0.60-0.65 and adjust based on win rate:
- Win rate > 60%? Lower confidence to increase frequency
- Win rate < 45%? Raise confidence to increase quality

### ADX Threshold

Controls minimum trend strength required for entry:

| Setting | ADX | Market Condition | Use When |
|---------|-----|------------------|----------|
| Strong Trends Only | 30+ | Strong directional | Bull/bear markets |
| Moderate Trends | 20-25 | Mixed conditions | Normal markets |
| Any Condition | 15-20 | Including ranges | Scalping/any market |
| No Filter | 0 | All conditions | Strategy has own filters |

**Note:** Some strategies (like VWAP) have internal ADX filters. Check strategy documentation.

### Stop Loss & Profit Target

Must align with how your model was trained:

| Profile | Stop ATR | Target ATR | Risk/Reward | Trade Style |
|---------|----------|------------|-------------|-------------|
| Scalper | 1.0-1.5 | 1.5-2.0 | 1:1-1.5 | Quick in/out |
| Day Trader | 1.5-2.0 | 2.0-3.0 | 1:1.5-2 | Intraday swings |
| Swing | 2.0-2.5 | 3.0-4.0 | 1:1.5-2 | Multi-hour holds |
| Conservative | 2.5-3.0 | 2.0-2.5 | 1:0.8-1 | Capital preservation |

⚠️ **Critical:** Your `target_atr` should match what the model was trained on. If the model was trained with 2.0 ATR targets, don't use 4.0 in live trading.

### Trailing Stops

Enable trailing stops to lock in profits:

```yaml
enable_trailing_stop: true
```

**When to Use:**
- ✅ Trending markets (let winners run)
- ✅ Breakout strategies (capture momentum)
- ✅ When unsure about target distance

**When NOT to Use:**
- ❌ Mean reversion strategies (may exit too early)
- ❌ Ranging markets (whipsaws)
- ❌ When you want consistent R:R ratios

---

## Troubleshooting

### Connection Issues

**Problem:** Bot won't connect to TopstepX

**Solutions:**
- Verify username and API key are correct
- Check internet connection is stable
- Confirm TopstepX services are operational
- Verify `market_hub` and `base_url` are correct
- Check if your IP is whitelisted (if required)

### File Not Found Errors

**Problem:** `Model file not found` or `Scaler not found`

**Solutions:**
```bash
# Use absolute paths in config
model: "/full/path/to/models/model.onnx"
scaler: "/full/path/to/models/scaler.pkl"

# Or verify relative paths
ls -la models/  # Check files exist

# Check file permissions
chmod 644 models/*.onnx models/*.pkl
```

### Strategy Errors

**Problem:** `Unknown strategy: 'my_strategy'`

**Solutions:**
- Check spelling: `3min_pivot_reversal` not `3min-pivot-reversal`
- Verify strategy is registered in `strategy_factory.py`
- List available strategies:
  ```bash
  python algoTrader.py --help
  ```

### Scaler/Contract Mismatch

**Problem:** `Scaler for 'NQ' not found`

**Solutions:**
- Check contract symbol parsing in logs
- Verify scaler file contains correct keys:
  ```python
  import pickle
  with open('models/scaler.pkl', 'rb') as f:
      scalers = pickle.load(f)
      print(scalers.keys())  # Should show ['ES', 'NQ', 'YM', 'RTY']
  ```
- Re-train scaler if necessary

### No Trades Executing

**Problem:** Bot runs but never enters trades

**Solutions:**
1. **Lower confidence threshold:**
   ```yaml
   entry_conf: 0.50  # Try lower value
   ```

2. **Lower ADX threshold:**
   ```yaml
   adx_thresh: 15  # Or 0 to disable
   ```

3. **Check market hours:**
   - Avoid trading during Asian session (low volume)
   - Best hours: 9:30 AM - 4:00 PM ET

4. **Verify model predictions:**
   - Add logging in `predict()` method
   - Check if model is loading correctly
   - Verify features are being calculated

5. **Check strategy-specific filters:**
   - VWAP: Has internal ADX 20-40 filter
   - Trend Pullback: Requires pullback trigger
   - Review strategy documentation

### Too Many Losses

**Problem:** High frequency of losing trades

**Solutions:**
1. **Increase confidence:**
   ```yaml
   entry_conf: 0.70  # Higher quality signals
   ```

2. **Increase ADX:**
   ```yaml
   adx_thresh: 25  # Stronger trends only
   ```

3. **Verify model/scaler match:**
   - Ensure you're using the correct model for the strategy
   - Check that scaler was trained with same features

4. **Paper trade first:**
   - Test with paper account for 2-4 weeks
   - Compare results to backtest expectations
   - Adjust parameters based on performance

5. **Check market conditions:**
   - Some strategies work better in trending vs ranging markets
   - Consider using different strategies for different conditions

### YAML Config Errors

**Problem:** Config file not loading correctly

**Solutions:**
```bash
# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('configs/my_config.yaml'))"

# Check for common issues:
# - Proper indentation (2 or 4 spaces, not tabs)
# - Quotes around strings with special characters
# - Boolean values: true/false (lowercase)
# - Numbers without quotes

# Example of correct YAML:
username: "MyUser"      # Quoted string
size: 1                 # Unquoted number
entry_conf: 0.60        # Unquoted float
enable_trailing_stop: false  # Lowercase boolean
```

---

## Performance Monitoring

### Key Metrics to Track

**Real-time Monitoring:**
- ✅ Entry signals and confidence levels
- ✅ Exit reasons (stop/target/time)
- ✅ Current ADX, ATR values
- ✅ P&L per trade

**Daily Review:**
- Win Rate (target: 50-60%)
- Profit Factor (target: >1.5)
- Average Win vs Average Loss (win should be larger or equal)
- Maximum Drawdown (consecutive losses)
- Trade Frequency (5-15 setups/week typical)

**Weekly Review:**
- Sharpe Ratio
- Maximum Adverse Excursion (MAE)
- Maximum Favorable Excursion (MFE)
- Best/worst trading days

### Logging

The bot logs all activity to `bot_log.log`. Monitor for:

```bash
# Tail logs in real-time
tail -f bot_log.log

# Search for entries
grep "ENTER LONG\|ENTER SHORT" bot_log.log

# Search for exits
grep "EXIT" bot_log.log

# Check errors
grep "ERROR\|Exception" bot_log.log
```

### Performance Dashboard (Optional)

Consider building a simple dashboard to track:
- Daily P&L chart
- Win rate by strategy
- Entry confidence distribution
- ADX distribution at entries
- Hourly performance heatmap

---

## Best Practices

### Before Going Live

1. ✅ **Paper trade minimum 2-4 weeks**
   - Track all trades in a spreadsheet
   - Calculate metrics (win rate, profit factor, etc.)
   - Compare to backtest expectations

2. ✅ **Understand your strategy deeply**
   - Know what features it uses
   - Understand entry/exit logic
   - Know what market conditions it prefers

3. ✅ **Test different parameter sets**
   - Try conservative, balanced, aggressive configs
   - Document which works best
   - Stick to parameters that match model training

4. ✅ **Have a risk management plan**
   - Maximum daily loss limit
   - Maximum number of trades per day
   - Rules for when to stop trading

5. ✅ **Prepare for technical issues**
   - Test internet failover
   - Have backup power
   - Know how to manually close positions

### Risk Management Rules

**Position Sizing:**
```yaml
# Start small!
size: 1  # One contract only until profitable

# Scale up gradually
# Month 1-2: 1 contract
# Month 3-4: 2 contracts (if profitable)
# Month 5+: 3+ contracts (if consistently profitable)
```

**Daily Limits:**
- 💰 Max daily loss: 2-3% of account
- 📊 Max number of trades: 5-10 per day
- 🛑 Stop after 3 consecutive losses
- ⏰ Avoid trading during major news events

**Time-based Rules:**
- 🌅 Best hours: 9:30 AM - 3:00 PM ET
- 🌙 Avoid: Asian session (8 PM - 2 AM ET)
- 📰 Avoid: 30 min before/after FOMC, NFP, CPI

**Position Management:**
- Never add to losing positions
- Consider partial exits at 1:1 R/R
- Use trailing stops in strong trends
- Close all positions before weekend (optional)

### System Requirements

**Hardware:**
- 💻 Stable computer/VPS (must stay on during market hours)
- 🔌 Backup power supply (UPS recommended)
- 📶 Reliable internet (>10 Mbps, backup connection recommended)

**Software:**
- 🐍 Python 3.10+ (latest patch version)
- 📦 All dependencies up to date
- 🔐 API credentials secured (not in code)

**Monitoring:**
- 📱 Mobile alerts for order fills (optional but recommended)
- 📧 Email alerts for errors (optional)
- 📊 Performance tracking spreadsheet

### Operational Checklist

**Daily Pre-Market:**
- [ ] Check bot_log.log for errors from previous session
- [ ] Verify API credentials are valid
- [ ] Confirm internet connection is stable
- [ ] Review today's economic calendar
- [ ] Check account balance and margin

**During Market:**
- [ ] Monitor logs in real-time (`tail -f bot_log.log`)
- [ ] Verify trades are executing as expected
- [ ] Check for any errors or warnings
- [ ] Stay near computer (or have mobile alerts)

**End of Day:**
- [ ] Review all trades executed
- [ ] Calculate daily P&L
- [ ] Update performance tracking
- [ ] Check for any anomalies
- [ ] Archive logs

**Weekly:**
- [ ] Review strategy performance metrics
- [ ] Compare actual vs expected performance
- [ ] Adjust parameters if needed (document changes)
- [ ] Backup configuration files
- [ ] Update models if retrained

---

## Advanced Topics

### Running Multiple Strategies

Run different strategies on different contracts:

**Terminal 1:**
```bash
python algoTrader.py --config configs/vwap_nq.yaml
```

**Terminal 2:**
```bash
python algoTrader.py --config configs/trend_es.yaml
```

**Terminal 3:**
```bash
python algoTrader.py --config configs/pivot_rty.yaml
```

**Best Practices:**
- Use different log files per bot
- Monitor all terminals simultaneously
- Keep total risk across all bots within limits
- Ensure strategies are uncorrelated

### Model Retraining

When to retrain your models:
- ✅ Performance degrades over time
- ✅ Market regime changes significantly
- ✅ You want to add new features
- ✅ Every 3-6 months (regular maintenance)

**Retraining Workflow:**
1. Collect new training data
2. Retrain model with updated data
3. Export to ONNX format
4. Backtest thoroughly
5. Paper trade 2-4 weeks
6. Deploy if performance improves

### API Rate Limits

Be aware of API rate limits:
- REST API: Typically 100 requests/minute
- SignalR: Unlimited tick data
- Historical data: May have daily limits

**Best Practices:**
- Cache historical data
- Don't make unnecessary API calls
- Implement exponential backoff on errors
- Monitor rate limit headers

### Custom Broker Integration

To use a different broker (must support ProjectX API):

1. **Update URLs in config:**
   ```yaml
   market_hub: "https://your-broker-rtc.com/hubs/market"
   base_url: "https://your-broker-api.com/api"
   ```

2. **Verify API compatibility:**
   - Check authentication method
   - Verify contract ID format
   - Test order placement format
   - Confirm historical data format

3. **Update `bot_utils.py` if needed:**
   - Modify authentication function
   - Adjust contract parsing
   - Update any broker-specific logic

---

## Files Reference

| File | Purpose |
|------|---------|
| `algoTrader.py` | Main entry point, argument parsing, initialization |
| `trading_bot.py` | Core bot engine (ticks, bars, orders, positions) |
| `config_loader.py` | YAML config file loading and validation |
| `bot_utils.py` | Authentication, logging, contract parsing |
| `strategy_base.py` | Abstract base class for all strategies |
| `strategy_factory.py` | Strategy registry and creation |
| `strategy_squeeze.py` | Squeeze V3 strategy implementation |
| `strategy_pivot_reversal_3min.py` | 3-minute pivot reversal |
| `strategy_pivot_reversal_5min.py` | 5-minute pivot reversal |
| `strategy_vwap_3min.py` | VWAP mean reversion |
| `strategy_trend_pullback.py` | Trend pullback V1 |
| `strategy_trend_pullback2.py` | Trend pullback V2 |
| `README.md` | This file |

---

## Support & Resources

### Getting Help

**For bot issues:**
1. Check the Troubleshooting section above
2. Review logs in `bot_log.log`
3. Enable debug logging: `logging.DEBUG` in code
4. Check TopstepX/ProjectX API status

**For strategy questions:**
1. Review strategy source code
2. Check feature definitions in `get_feature_columns()`
3. Review `add_features()` for calculations
4. Understand `should_enter_trade()` logic

### Common Questions

**Q: Can I run multiple strategies simultaneously?**
A: Yes! Run separate instances with different configs. Keep total risk across all bots within your limits.

**Q: How do I update a strategy?**
A: Simply modify the strategy file. The bot will use the updated logic on next restart. No need to change core bot code.

**Q: Can I use my own custom model?**
A: Absolutely! Create a custom strategy class, train your model, export to ONNX format, and register it in the factory.

**Q: What if my broker isn't TopstepX?**
A: If your broker uses the ProjectX API framework, just update the `market_hub` and `base_url` in your config.

**Q: How do I backtest strategies?**
A: This is a live trading bot. For backtesting, you'll need separate backtesting code that uses historical data and the same strategy logic.

**Q: Can I use this for crypto or stocks?**
A: Not without significant modifications. The bot is designed specifically for futures contracts via the ProjectX API.

---

## Version History

- **v1.0** (Original): Single hardcoded strategy
- **v2.0** (Refactor): Pluggable strategy architecture
  - Added `BaseStrategy` abstract class
  - Implemented `StrategyFactory`
  - Separated trading logic from AI logic
- **v2.1** (Current): Enhanced features
  - Added YAML configuration support
  - Six production strategies included
  - Improved error handling
  - Better logging and monitoring
  - Comprehensive documentation

---

## License & Warranty

This software is provided "as is" without warranty of any kind, express or implied. The author and contributors are not responsible for any financial losses incurred through the use of this software.

**USE AT YOUR OWN RISK.**

Trading futures and derivatives involves substantial risk of loss and is not suitable for all investors. Past performance is not indicative of future results. You should carefully consider whether trading is suitable for you in light of your circumstances, knowledge, and financial resources.

Only trade with risk capital – money you can afford to lose.

---

## Acknowledgments

Thanks to all contributors and the trading community for feedback and improvements.

Special thanks to the developers of:
- **pandas-ta** - Technical analysis library
- **ONNX Runtime** - Model inference
- **SignalR** - Real-time data streaming

---

**Happy Trading! 🚀**

*"The best trade is often the one you don't take. Always prioritize capital preservation over profit maximization."*

---

## Quick Reference

### Start Bot with Config
```bash
python algoTrader.py --config configs/my_strategy.yaml
```

### View Available Strategies
```bash
python algoTrader.py --help
```

### Monitor Logs
```bash
tail -f bot_log.log
```

### Stop Bot
Press **Ctrl+C**

### Emergency Position Close
Log into TopstepX web interface and manually close positions if bot fails.

---

## Contact & Contributions

For bug reports, feature requests, or contributions, please follow your project's contribution guidelines.

Remember: **Always test thoroughly on paper accounts before risking real capital!**
