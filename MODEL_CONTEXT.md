# CISD+OTE Hybrid Model Reference

## Current Production Model
- **Version**: v5.5
- **File**: `cisd_ote_hybrid_v5_5_F4.onnx`
- **Checkpoint**: `cisd_hybrid_v5_5_F4_signal_f1.pth`
- **Strategy**: ICT/SMC — CISD + OTE (Fib 0.618–0.786 retracement zone)
- **Instruments**: ES, NQ, RTY, YM, GC (micros: MES, MNQ, MRTY, MYM, MGC)
- **Bar timeframe**: 5-minute

---

## ONNX Interface

### Inputs
| Name | Shape | dtype |
|------|-------|-------|
| `features` | [B, 64, 42] | float32 |
| `cisd_features` | [B, 32] | float32 |
| `time_of_day` | [B, 64] | float32 |
| `day_of_week` | [B, 64] | int64 |
| `instrument_ids` | [B] | int64 |
| `session_ids` | [B, 64] | int64 |

### Outputs
| Name | Shape | Notes |
|------|-------|-------|
| `signal_probs` | [B, 2] | `[:,1]` = signal probability — threshold on this |
| `confidence` | [B] | 0–1 calibrated confidence |
| `risk` | [B, 1] | Predicted R:R (informational only) |

### Entry Logic
```python
signal_prob = signal_probs[0, 1]
direction   = 'BUY' if cisd_features[0, 4] > 0 else 'SELL'  # idx 4 = zone_is_bullish

# Thresholds
# >=0.85 -> standard entry       (PF 6.06, 37 trades/backtest, AvgRR 4.62)
# >=0.90 -> 4R target sizing     (PF 6.56, 21 trades/backtest, AvgRR 4.92)
# >=0.70 -> forward test only    (PF 3.07 -- used for v5.1 comparison)
```

---

## CISD Feature Vector — 32 Indices (exact order required)

```
 0  zone_height_vs_atr          Zone height / ATR
 1  price_vs_zone_top           (close - zone_top) / zone_height
 2  price_vs_zone_bot           (close - zone_bot) / zone_height
 3  zone_age_bars               Bars since zone created / 40
 4  zone_is_bullish             +1.0 = BUY, -1.0 = SELL  <- direction flag
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
--- v5.3+ regime/vol context (do not remove or reorder) ---
28  atr_compression_ratio       ATR / 50-bar ATR mean  (<0.70 = low-vol)
29  htf_trend_strength          Consecutive bars HTF trend held / 96
30  delta_momentum_ratio        5-bar / 20-bar abs-delta SMA
31  regime_transition_recency   1.0 at HTF flip, decays to 0 over 48 bars
```

**Rules**: clip all values to [-10, 10], NaN -> 0.0.

---

## Instrument Mapping

```python
INSTRUMENT_IDS = {
    'ES': 0, 'NQ': 1, 'RTY': 2, 'YM': 3, 'GC': 4,
    # Micros use same ID as full-size counterpart
    'MES': 0, 'MNQ': 1, 'MRTY': 2, 'MYM': 3, 'MGC': 4,
}

POINT_VALUES = {
    'ES': 50.0,  'NQ': 20.0,  'RTY': 10.0,  'YM': 5.0,   'GC': 100.0,
    'MES': 5.0,  'MNQ': 2.0,  'MRTY': 5.0,  'MYM': 0.50, 'MGC': 10.0,
}
```

---

## Architecture (read-only — do not modify)

```
FFMBackbone(256-dim)     pre-trained, 66% of layers frozen
  |
cisd_projection          Linear(32->64) -> GELU -> Dropout -> Linear(64->64)
  | concat with backbone output
fusion                   Linear(320->256) -> GELU -> LayerNorm -> Dropout
  |
signal_head              Linear(256->128) -> GELU -> Dropout -> Linear(128->2)
risk_head                Linear(256->128) -> GELU -> Dropout -> Linear(128->1) -> Softplus
confidence_head          Linear(256->64)  -> GELU -> Dropout -> Linear(64->1)  -> Sigmoid
```

The backbone is a pre-trained Futures Foundation Model (FFM). Never retrain it — fine-tune heads only.

---

## Labeler Rules (important for any label generation code)

Two conditioned RR thresholds are applied at label time. **Both must be preserved** if labeling code is ever modified.

**1. Vol-conditioned minimum RR (v5.4+)**
```python
if atr_compression < 0.70:   effective_min_rr = 2.0   # low vol -- barriers not reached
elif atr_compression < 0.85: effective_min_rr = 1.5   # mid vol
else:                         effective_min_rr = 1.0   # normal (standard)
```

**2. Trend-conditioned minimum RR (v5.5) — asymmetric, longs only**
```python
strong_trend  = trend_run_bars >= 96              # full session of one direction
counter_trend = is_long and htf_trend == -1       # longs vs downtrend ONLY
# shorts vs uptrend are NOT gated (they perform well historically)
if strong_trend and counter_trend:
    trend_min_rr = 2.0
```

**Final threshold**: `max(vol_min_rr, trend_min_rr)`

These thresholds exist because v5.1 trained on low-vol and counter-trend noise labels,
causing live trading failures in compressed markets and sustained downtrends.
Removing them reverts to v5.1 failure modes.

---

## Changes from v5.1 to v5.5 (and why)

The bot currently runs v5.1. v5.5 is the new production model. These are the
differences between the existing bot code and v5.5.

### v5.3 — Added 4 regime/vol context features (indices 28-31)
**What changed**: `cisd_features` vector grew from 28 to 32 elements.
**Why**: v5.1 had no way to encode market regime at inference time. The model
could not distinguish a normal setup from one occurring in low-vol compression
or immediately after a trend flip -- both confirmed live failure conditions.
The 4 new features inject this context into the CISD projection layer so the
model can suppress these conditions at inference.
**Code impact**: anywhere `cisd_features` is assembled it must now produce 32
values not 28. The 4 new features must appear last, in index order.

### v5.4 — Labeler quality gates (vol-conditioned RR thresholds)
**What changed**: labeler requires higher minimum RR in low-vol conditions.
ATR compression < 0.70 requires 2R, < 0.85 requires 1.5R, normal stays 1R.
**Why**: v5.1 trained on signals that barely reached 1R in compressed ATR
conditions. These labeled positive but failed live because the market lacked
range to sustain the move before session end. Removing this noise from training
teaches the model not to fire in low-vol regimes.
**Code impact**: labeler only -- no inference code changes.

### v5.5 — Asymmetric counter-trend gate
**What changed**: labeler requires 2R minimum for bullish CISDs when HTF
downtrend has held >= 96 bars. Bearish CISDs in uptrends are NOT gated.
**Why**: v5.1 trained on bullish CISDs that hit a 1R relief bounce then failed
as the downtrend reasserted -- the most common live loss pattern. The gate is
asymmetric (longs only) because short setups in uptrends perform well
historically. Gating both directions (tried in v5.4) over-removed bear signals
and made the model long-biased. 96 bars (one full session) is the gate because
shorter-duration trends are normal market structure, not the failure condition.
**Code impact**: labeler only -- no inference code changes.

### Net inference code changes v5.1 -> v5.5
Only one change affects the bot: `cisd_features` must be 32 elements not 28.
The 4 new features (indices 28-31) must be computed and appended in order:

```python
# Append to existing 28-feature vector before inference
features[28] = atr / rolling_atr_50bar_mean                    # atr_compression_ratio
features[29] = min(htf_trend_run_bars / 96.0, 1.0)             # htf_trend_strength
features[30] = abs_delta_5bar_sma / abs_delta_20bar_sma        # delta_momentum_ratio
features[31] = max(0.0, 1.0 - (bars_since_htf_flip / 48.0))   # regime_transition_recency

# Clip all 32 features to [-10, 10], NaN -> 0.0
```

Entry threshold also shifts: v5.1 used >=0.70 for standard entry.
v5.5 uses >=0.85 for standard entry, >=0.90 for 4R sizing.
>=0.70 is kept only for forward-test comparison against v5.1.

---

## Walk-Forward Results Summary

Test set: 20,680 bars across 4 folds (2022-2025 out-of-sample).

| Version | >=0.70 PF | >=0.85 PF | >=0.90 PF | >=0.90 Prec |
|---------|-----------|-----------|-----------|-------------|
| v5.1    | 4.22      | 4.74      | 8.71      | 0.682       |
| v5.3    | 3.54      | 5.64      | 6.42      | 0.586       |
| v5.4    | 3.06      | 4.50      | 4.70      | 0.500       |
| v5.5    | 3.07      | 6.06      | 6.56      | 0.571       |

v5.5 >=0.85 (PF 6.06) is the recommended live entry threshold.
v5.5 >=0.90 (PF 6.56, AvgRR 4.92) is used for 4R target sizing.

---

## Known Failure Modes (suppressed in v5.5)

1. **Low ATR compression** (`atr_compression_ratio < 0.70`): market lacks range to
   reach RR barriers before session close. Feature idx 28 gives the model this signal.

2. **Bullish CISD counter to strong downtrend**: when HTF trend has been -1 for 8+
   hours (96 bars), longs fail as trend reasserts after a brief bounce. Feature idx 29
   (`htf_trend_strength`) encodes trend maturity.

3. **Post-HTF-flip period**: first ~48 bars after a trend direction change. Feature
   idx 31 (`regime_transition_recency`) decays 1.0 -> 0 over this window.