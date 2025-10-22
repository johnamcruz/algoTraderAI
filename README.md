# AI Algo Trading Bot

## Overview

This Python script implements a **real-time algorithmic trading bot** designed for futures markets (specifically tested on ES, NQ, YM, RTY via TopstepX platform). It leverages a machine learning model (LSTM with Attention) trained to predict profitable trend-following breakouts from periods of low volatility (TTM Squeeze).

### Core Strategy Components

The bot operates through the following workflow:

1. **Real-time Data**: Connects to the TopstepX SignalR feed for live trade ticks
2. **Bar Aggregation**: Aggregates incoming ticks into time-based bars (e.g., 5-minute)
3. **Feature Calculation**: Calculates technical indicators and specific features related to the TTM Squeeze and potential breakouts using `pandas-ta`
4. **AI Prediction**: Feeds the latest sequence of features (scaled) into a pre-trained ONNX model to predict the probability of a profitable upward or downward breakout
5. **Signal Generation**: Combines the AI prediction with filter conditions (ADX, Volume Surge, Squeeze ON) to generate entry signals on bar close
6. **Trade Management**: Enters trades based on signals and manages exits using tick-based ATR stop-loss and profit targets derived from the AI model's training objective

---

## ⚠️ Disclaimer

**This is a trading bot implementation. Trading futures involves substantial risk of loss and is not suitable for all investors.**

- Use this code **at your own risk**
- Past performance is **not indicative of future results**
- Ensure you understand the code and risks before deploying with real capital
- **Paper trading is highly recommended** before going live

---

## Strategy Logic

The bot aims to capture explosive trend moves that often follow periods of market consolidation, identified by the **TTM Squeeze indicator**.

### Entry Signal (on Bar Close)

All of the following conditions must be met:

- ✅ **TTM Squeeze** must be active (`squeeze_on == 1`)
- ✅ **Trend strength** must be sufficient (`ADX > adx_thresh`)
- ✅ **Volume** must confirm potential breakout (`vol_surge == 1`)
- ✅ **AI model confidence** for the predicted direction (UP or DOWN) must exceed the `entry_conf` threshold

### Exit Logic (Checked on Every Tick)

The bot exits positions when:

- **Stop Loss**: Price touches the initial stop-loss level, calculated as:
  ```
  entry_price ± (entry_atr × stop_atr_mult)
  ```

- **Profit Target**: Price touches the profit target level, calculated as:
  ```
  entry_price ± (entry_atr × target_atr_mult)
  ```

**Note**: The `target_atr_mult` should match the R/R target the loaded AI model was trained on.

> *AI Reversal exit logic from the backtester is not implemented in this version but could be added.*

---

## AI Model Details

The bot uses a pre-trained neural network saved in the **ONNX format**. The model architecture is an LSTM with an Attention mechanism.

### Input Specification

- **Sequence Length**: Last 60 time steps (bars)
- **Features per Bar**: 9 engineered features

### Features Used

| Feature | Description |
|---------|-------------|
| `compression_level` | Tightness of Bollinger Bands vs Keltner Channels |
| `squeeze_duration` | Number of bars the squeeze has been active |
| `bb_expanding` | Is Bollinger Band width increasing? |
| `atr_expanding` | Is ATR increasing? |
| `price_in_range` | Price position within Bollinger Bands (0-1) |
| `rsi` | Standard RSI(14) |
| `compressed_momentum` | ROC(10) only during a squeeze |
| `vol_surge` | Volume significantly above its moving average? |
| `body_strength` | Candle body size relative to ATR |

### Architecture

```
Input: [Batch, 60 timesteps, 9 features]
    ↓
LSTM Layer (hidden_size=32)
    ↓
Linear Attention Layer
    ↓
Attention Weighting (softmax)
    ↓
Context Vector (weighted sum)
    ↓
Dropout (p=0.5)
    ↓
FC Layer 1: Linear(32 → 16) + ReLU
    ↓
Dropout (p=0.5)
    ↓
FC Layer 2: Linear(16 → 2)
    ↓
Output: [DOWN_logit, UP_logit]
```

### Model Output

The bot converts the model's output logits into **probabilities** (confidence scores) using a Softmax function:

```
confidence_DOWN = softmax(logits)[0]
confidence_UP = softmax(logits)[1]
```

### Training Goal

The model was trained to predict whether a squeeze setup would result in:

1. Price hitting a specific **ATR-based profit target** before hitting an ATR-based stop loss
2. Confirmation by a **Supertrend flip** in the same direction
3. Within a **20-bar lookahead window**

---

## Requirements

### Python Version

- **Python 3.10+** (ideally matching the version used for training)
- Managed via `pyenv` is recommended

### Dependencies

A virtual environment is **strongly recommended**.

**Required packages:**
```bash
pip install onnxruntime pandas pandas-ta signalrcore requests numpy scikit-learn python-dateutil
```

---

## Setup

### 1. Clone/Download

Get the `algotrader_ai.py` script.

### 2. Environment Setup

Navigate to the script's directory in your terminal:

```bash
cd /path/to/script
```

Create a Python virtual environment:
```bash
python3 -m venv venv
```

Activate it:
- **macOS/Linux**: `source venv/bin/activate`
- **Windows**: `.\venv\Scripts\activate`

### 3. Install Packages

```bash
pip install onnxruntime pandas pandas-ta signalrcore requests numpy scikit-learn python-dateutil
```

### 4. API Credentials

Obtain your **TopstepX** username and API key.

⚠️ **Do not hardcode these in the script** - use command-line arguments.

### 5. Model & Scaler Files

You need two files generated from your Colab training script:

1. **ONNX model file** (e.g., `SUPERTRADER_strategy_3_runX.onnx`)
2. **Pickled scaler file** (e.g., `SUPERTRADER_scalers_strategy_3_runX.pkl`)

Place these files in a location accessible by the script (e.g., a `models` subfolder).

---

## Usage

Run the bot from your activated virtual environment using command-line arguments.

### Basic Command Structure

```bash
python algotrader_ai.py --account <YOUR_ACCOUNT_ID> \
                        --contract <FULL_CONTRACT_ID> \
                        --size <TRADE_SIZE> \
                        --username <YOUR_USERNAME> \
                        --apikey <YOUR_API_KEY> \
                        --timeframe <BAR_TIMEFRAME> \
                        --model <PATH_TO_ONNX_MODEL> \
                        --scaler <PATH_TO_SCALER_PKL> \
                        --entry_conf <CONFIDENCE_THRESHOLD> \
                        --adx_thresh <ADX_THRESHOLD> \
                        --stop_atr <STOP_ATR_MULTIPLIER> \
                        --target_atr <TARGET_ATR_MULTIPLIER> \
                        --ai_reversal <REVERSAL_CONFIDENCE>
```

### Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--account` | Your TopstepX account ID | `TS001234SIM` |
| `--contract` | The full TopstepX contract ID | `CON.F.US.RTY.Z25` |
| `--size` | Number of contracts per trade | `1` |
| `--username` | Your TopstepX login username | `MyUser` |
| `--apikey` | Your TopstepX API key | `MySecretKey` |
| `--timeframe` | Bar aggregation interval in minutes (1, 3, or 5) | `5` (default) |
| `--model` | **Required.** Path to your trained `.onnx` model file | `models/model.onnx` |
| `--scaler` | **Required.** Path to your corresponding `.pkl` scaler file | `models/scaler.pkl` |
| `--entry_conf` | Minimum AI confidence (0.0 to 1.0) required to enter a trade | `0.55` |
| `--adx_thresh` | Minimum ADX value required to enter | `25` |
| `--stop_atr` | Stop loss multiplier (times entry bar ATR) | `2.0` |
| `--target_atr` | Profit target multiplier (times entry bar ATR). **Must match training target!** | `3.0` |
| `--ai_reversal` | AI confidence for opposite signal to trigger an exit (0.0 to 1.0) | `0.65` (default) |

### Example: Running RTY Strategy

Based on successful backtest parameters (Strategy 3, Run 4):

```bash
python algotrader_ai.py \
    --account TS001234SIM \
    --contract CON.F.US.RTY.Z25 \
    --size 1 \
    --username MyUser \
    --apikey MySecretKey \
    --timeframe 5 \
    --model "models/SUPERTRADER_strategy_3_run4_logicfix.onnx" \
    --scaler "models/SUPERTRADER_scalers_strategy_3_run4_logicfix.pkl" \
    --entry_conf 0.55 \
    --adx_thresh 25 \
    --stop_atr 2.0 \
    --target_atr 3.0
```

### Example: Running ES Strategy

For conservative settings on ES:

```bash
python algotrader_ai.py \
    --account TS001234SIM \
    --contract CON.F.US.ES.Z25 \
    --size 1 \
    --username MyUser \
    --apikey MySecretKey \
    --timeframe 5 \
    --model "models/ES_strategy_1.onnx" \
    --scaler "models/ES_scaler_1.pkl" \
    --entry_conf 0.60 \
    --adx_thresh 20 \
    --stop_atr 2.5 \
    --target_atr 3.0
```

### Stopping the Bot

Press **Ctrl+C** to stop the bot gracefully.

---

## Important Notes

### Critical Parameter Alignment

⚠️ **The `--target_atr` parameter MUST match the target used during model training!**

If your model was trained with:
- `target_atr_mult = 3.0` → Use `--target_atr 3.0`
- `target_atr_mult = 2.5` → Use `--target_atr 2.5`
- `target_atr_mult = 2.0` → Use `--target_atr 2.0`

Misalignment will cause the bot to use incorrect profit targets.

### Recommended Parameter Ranges

Based on backtest results:

| Parameter | Conservative | Balanced | Aggressive |
|-----------|--------------|----------|------------|
| `entry_conf` | 0.60-0.65 | 0.55-0.60 | 0.50-0.55 |
| `adx_thresh` | 20-25 | 15-20 | 10-15 |
| `stop_atr` | 2.0-2.5 | 1.5-2.0 | 1.0-1.5 |
| `target_atr` | 2.0-2.5 | 2.5-3.0 | 3.0-4.0 |

---

## Troubleshooting

### Common Issues

**"Cannot connect to TopstepX"**
- Verify your username and API key are correct
- Check your internet connection
- Ensure TopstepX services are operational

**"Model file not found"**
- Verify the path to your `.onnx` file is correct
- Use absolute paths if relative paths fail
- Check file permissions

**"Scaler file not found"**
- Verify the path to your `.pkl` file is correct
- Ensure the scaler corresponds to the correct model
- Check file permissions

**"No trades being placed"**
- Lower `--entry_conf` threshold (try 0.50)
- Lower `--adx_thresh` threshold (try 15)
- Verify squeeze conditions are occurring in the market
- Check if volume surge condition is too restrictive

**"Too many losing trades"**
- Increase `--entry_conf` threshold (try 0.65)
- Increase `--adx_thresh` threshold (try 25)
- Review if `--target_atr` matches model training
- Consider paper trading different parameter combinations

---

## Performance Monitoring

### Key Metrics to Track

While the bot is running, monitor:

- **Win Rate**: Should be 50-60% for healthy performance
- **Profit Factor**: Should be > 1.3 (ideally 1.5+)
- **Average Win vs Average Loss**: Avg Win should be > Avg Loss
- **Max Drawdown**: Track consecutive losses
- **Trade Frequency**: Should see 5-10 setups per week per ticker

### Logging

The bot logs all actions. Review logs regularly:
- Entry signals and reasons
- Exit signals and P&L
- Model predictions and confidence levels
- Technical indicator values at entry

---

## Next Steps

1. **Paper Trade First**: Run on simulation account for at least 2-4 weeks
2. **Track Performance**: Compare live results to backtest expectations
3. **Parameter Tuning**: Adjust confidence/ADX thresholds based on live results
4. **Multiple Tickers**: Consider running multiple instances on different contracts
5. **Walk-Forward Testing**: Retrain model monthly on recent data

---

## License & Warranty

This software is provided "as is" without warranty of any kind. The authors are not responsible for any financial losses incurred through use of this software.

Trading futures and options involves substantial risk of loss and is not appropriate for all investors. Only risk capital should be used for trading.

---

## Version History

- **v1.0**: Initial release with LSTM + Attention model
- **v1.1**: Added AI reversal exit logic (optional)
- **v1.2**: Improved tick-based exit handling
- **v2.0**: Multi-timeframe support (1, 3, 5 minute bars)

---

**Happy Trading! 🚀**

*Remember: The best trade is often the one you don't take. Always prioritize capital preservation.*
