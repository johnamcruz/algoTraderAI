# CISD+OTE Model Reference

## Current Production Model — v7

- **File**: `models/cisd_ote_hybrid_v7.onnx`
- **Strategy class**: `CISDOTEStrategyV7` (`cisd-ote7`)
- **Instruments**: MES, MNQ, MGC (and full-size equivalents: ES, NQ, GC)
- **Bar timeframe**: 5-minute
- **Recommended entry_conf**: `0.80` (moderate) | `0.90` (conservative)
- **Required min_risk_rr**: `2.0` (skip trades where model predicts < 2R)

---

## v7 ONNX Interface

### Inputs

| Name | Shape | dtype | Notes |
|------|-------|-------|-------|
| `features` | [B, 96, 67] | float32 | FFM sequence — 96 bars × 67 derived features |
| `strategy_features` | [B, 10] | float32 | CISD zone features (see table below) |
| `candle_types` | [B, 96] | int64 | 0=doji, 1=bull-strong, 2=bear-strong, 3=bull-pin, 4=bear-pin, 5=neutral |
| `time_of_day` | [B, 96] | float32 | Minutes-since-midnight / 1440 |
| `day_of_week` | [B, 96] | int64 | 0=Mon … 4=Fri |
| `instrument_ids` | [B] | int64 | See instrument mapping below |
| `session_ids` | [B, 96] | int64 | 0=pre-market, 1=london, 2=ny_am, 3=ny_pm (ET) |

### Outputs

| Name | Shape | Notes |
|------|-------|-------|
| `signal_logits` | [B, 2] | Raw logits — apply softmax to get signal probabilities |
| `risk_predictions` | [B, 1] | Predicted R:R ratio — used for tier-snapped TP |
| `confidence` | [B] | `max(softmax(signal_logits))` — threshold on this for entry |

### Entry Logic

```python
confidence  = outputs['confidence'][0]          # threshold at 0.80
direction   = 'BUY' if cisd[4] > 0 else 'SELL' # cisd[4] = zone_is_bullish
predicted_rr = outputs['risk_predictions'][0]    # gates and drives TP tier

# Entry allowed when:
#   confidence >= entry_conf (e.g. 0.80)
#   predicted_rr >= min_risk_rr (e.g. 2.0)
```

---

## v7 CISD Feature Vector — 10 Indices (`strategy_features`)

```
Index  Name                    Description
─────────────────────────────────────────────────────────────────────
  0    zone_height_vs_atr      Zone height / ATR14  (zone geometry)
  1    price_vs_zone_top       (close - zone_top) / zone_height
  2    price_vs_zone_bot       (close - zone_bot) / zone_height
  3    zone_age_bars           Bars since zone created / 40
  4    zone_is_bullish         +1.0 = BUY, -1.0 = SELL  ← direction flag
  5    cisd_displacement_str   Displacement body ratio (quality of CISD candle)
  6    had_liquidity_sweep     1.0 if prior swing swept within 10 bars
  7    entry_distance_pct      Signed depth into OTE zone (negative = inside)
  8    risk_dollars_norm       Stop-loss size / 300 (risk normalisation)
  9    in_optimal_session      1.0 if 09:00–11:00 ET (NY prime window flag)
```

**Rules**: clip all values to [-10, 10], NaN → 0.0.

---

## v7 Architecture

```
FFMBackbone (256-dim)         pre-trained Futures Foundation Model
  Input: features [B, 96, 67]
  |
strategy_projection           Linear(10→64) → GELU → Dropout → Linear(64→64)
  Input: strategy_features [B, 10]
  | concat with backbone output
fusion                        Linear(320→256) → GELU → LayerNorm → Dropout
  |
signal_head                   Linear(256→128) → GELU → Dropout → Linear(128→2)
risk_head                     Linear(256→128) → GELU → Dropout → Linear(128→1) → Softplus
confidence_head               Linear(256→64)  → GELU → Dropout → Linear(64→1)  → Sigmoid
```

- **seq_len**: 96 bars (~8 hours of 5-min context)
- **FFM features**: 67 (computed by `futures_foundation.derive_features()`)
- **CISD features**: 10 (zone geometry + quality + session only)
- **Warmup**: ~300 bars before first prediction

The backbone is a pre-trained Futures Foundation Model. Fine-tune heads only — never retrain the backbone.

---

## Tier-Snapped Profit Target

The risk head predicts a continuous R:R. At inference, this is floor-snapped to calibrated tiers:

```python
raw_rr = risk_predictions[0]

if   raw_rr >= 4.0: rr = 4.0
elif raw_rr >= 3.0: rr = 3.0
elif raw_rr >= 2.0: rr = 2.0
elif raw_rr >= 1.5: rr = 1.5
else:               rr = max(raw_rr, 1.0)

target_pts = stop_pts * rr
```

**F4 validation hit rates (with min_risk_rr=2.0 gate):**
- ≥ 2R tier: 85% hit rate
- ≥ 3R tier: 81% hit rate
- ≥ 4R tier: 81% hit rate

**`high_conf_multiplier` is always disabled for `cisd-ote7`** — it would corrupt the risk head's calibrated targets. The `algoTrader.py` entry point enforces this.

---

## Instrument Mapping

```python
INSTRUMENT_MAP = {
    'ES': 0, 'NQ': 1, 'RTY': 2, 'YM': 3, 'GC': 4,
    # Micros resolved to parent before lookup
    'MES': 0, 'MNQ': 1, 'MGC': 4,
}

POINT_VALUES = {
    'ES': 50.0,  'NQ': 20.0,  'RTY': 10.0,  'YM': 5.0,   'GC': 100.0,
    'MES': 5.0,  'MNQ': 2.0,  'MRTY': 5.0,  'MYM': 0.50, 'MGC': 10.0,
}
```

---

## v7 vs v5.1 — Key Differences

| | v5.1 | v7 |
|---|---|---|
| **Strategy flag** | `cisd-ote` | `cisd-ote7` |
| **Model file** | `cisd_ote_hybrid_v5_1.onnx` | `cisd_ote_hybrid_v7.onnx` |
| **Sequence length** | 64 bars | 96 bars (~8 hrs context) |
| **FFM features** | 42 custom | 67 via `derive_features()` |
| **CISD features** | 32 (zone + market context) | 10 (zone geometry only) |
| **CISD input name** | `cisd_features` | `strategy_features` |
| **New input** | — | `candle_types` [B, 96] int64 |
| **Market context** | Encoded in 32 CISD features | Encoded in 67 FFM sequence |
| **Session gate** | Hard: 7am–4pm ET (training distribution) | None — model self-regulates via `in_optimal_session` feature |
| **Profit target** | Fixed 2R, extended to 4R at ≥90% conf | Tier-snapped: 1.5R / 2R / 3R / 4R from risk head |
| **high_conf_multiplier** | Supported | **Disabled** (corrupts tier-snapped TP) |
| **Entry gate: RR** | None | `min_risk_rr` — skip when predicted RR < threshold |
| **Confidence output** | Separate sigmoid head | `max(softmax(signal_logits))` |
| **Recommended conf** | 0.70–0.85 | 0.80 |
| **Signal frequency** | ~18 signals/scenario | ~40 signals/scenario (2× more signals) |
| **Regime gate** | `min_vty_regime` (0.75) | Built into backbone sequence |
| **OTE depth gate** | `min_entry_distance` (3.0) | Built into backbone sequence |

### Why v7 Generates More Signals

v5.1 filtered heavily at the feature level (volatility regime gate, OTE depth gate, session gate). v7 encodes this context into the 96-bar FFM backbone sequence, letting the model learn when these conditions matter rather than hard-blocking them. The result is ~2× signal frequency with similar or better precision at `conf >= 0.80`.

### v7 Known Limitation — Persistent Trend Blindness

The 96-bar context window (~8 hours) means v7 cannot see multi-day or weekly trend direction. In sustained bear markets (e.g. 2022 drawdown), the model fires bullish CISDs because the local structure looks valid — it has no awareness that the daily trend has been down for weeks. This is the primary driver of bear_2022 underperformance. A future version (v8) should add a daily HTF structure feature to address this.

---

## Walk-Forward Backtest Summary (v7, conf=0.80, min_risk_rr=2.0)

MNQ across 6 market-regime scenarios, $200 risk/trade, $400 MLL, $6,000 PT:

| Scenario | Period | P&L | PT Hit? | Notes |
|----------|--------|-----|---------|-------|
| `oos_2021` | 2021 | positive | ✅ | F1 trained on pre-2022 data — partially in-sample |
| `recovery_2023` | 2023 | positive | ✅ | Post-rate-hike rebound |
| `banking_2023` | Mar–May 2023 | positive | ✅ | SVB collapse, high-vol chop |
| `selloff_2024` | Jul–Sep 2024 | positive | ✅ | Sharp selloff + recovery |
| `bear_2022` | Jan–Oct 2022 | negative | ❌ | Persistent downtrend exceeds 8hr context |
| `recent_120d` | 2025 | varies | — | Most recent OOS window |

**4/6 profit targets hit** across scenarios, with losses contained well within MLL on the two misses.

---

## Legacy Reference — v5.1 CISD Feature Vector (32 indices)

Kept for historical reference and in case v5.1 is used for comparison runs.

```
 0  zone_height_vs_atr          Zone height / ATR
 1  price_vs_zone_top           (close - zone_top) / zone_height
 2  price_vs_zone_bot           (close - zone_bot) / zone_height
 3  zone_age_bars               Bars since zone created / 40
 4  zone_is_bullish             +1.0 = BUY, -1.0 = SELL
 5  cisd_displacement_strength  Retracement ratio (> 0.70)
 6  had_liquidity_sweep         1 if prior swing swept within 10 bars
 7  htf_trend_direction         60-min HTF: +1 up / 0 neutral / -1 down
 8  trend_alignment             1 = zone matches HTF trend
 9  rejection_wick_ratio        Directional wick / body
10  close_position              (close - low) / candle range
11  volume_trend                5-bar vol SMA / 20-bar vol SMA
12  cumulative_delta_ratio      Daily cumulative delta / volume
13  price_vs_ema20              (close - EMA20) / ATR
14  gap_from_prior_close        (close - prior close) / ATR
15  session_progress            Minutes since 9am ET / 420
16  day_of_week_feat            0=Mon ... 4=Fri
17  confluence_score            Confluence factor count / 10
18  risk_dollars_norm           Risk dollars / 300
19  in_optimal_session          1 if 9:00-11:00 ET
20  entry_distance_pct          Entry vs zone centre / zone height
21  ffm_sess_dist_from_vwap     FFM: distance from session VWAP
22  ffm_str_structure_state     FFM: market structure state
23  ffm_ret_acceleration        FFM: momentum acceleration x100
24  ffm_vty_atr_of_atr          FFM: ATR of ATR
25  ffm_sess_dist_from_open     FFM: distance from session open
26  ffm_ret_momentum_10         FFM: 10-bar momentum x100
27  ffm_vol_delta_proxy         FFM: volume delta proxy x0.001
--- v5.3+ regime/vol context ---
28  atr_compression_ratio       ATR / 50-bar ATR mean  (<0.70 = low-vol)
29  htf_trend_strength          Consecutive bars HTF trend held / 96
30  delta_momentum_ratio        5-bar / 20-bar abs-delta SMA
31  regime_transition_recency   1.0 at HTF flip, decays to 0 over 48 bars
```

v5.1 entry: `conf >= 0.70` standard, `>= 0.85` recommended live. Strategy flag: `cisd-ote`.
