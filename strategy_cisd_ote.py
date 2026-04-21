#!/usr/bin/env python3
"""
CISD+OTE Strategy Implementation (v5.1) — FFM Hybrid Transformer

Implements BaseStrategy for the CISD+OTE walk-forward model trained
against the Futures Foundation Model (FFM) backbone.

Key differences from other strategies:
  - Two ONNX inputs: 64-bar FFM sequence + 28-element point-in-time CISD vector
  - No scaler: FFM features are pre-normalized; CISD features clipped to [-10, 10]
  - Direction from zone_is_bullish (CISD feature idx 4), not from prediction class
  - Prediction returns: 0=Hold, 1=Buy, 2=Sell (mapped from zone direction)
  - Warmup: ~200 bars for stable HTF range / pivot detection

Walk-forward results (v5.1 @ ≥0.90 confidence):
  Combined: 22 trades | Precision 68.2% | PF 8.71 | AvgRR 4.07
  F2: 0.875 precision | F3: 0.600 precision | F4: 1.000 precision
"""

import logging
import numpy as np
import pandas as pd
import onnxruntime
from collections import deque
from typing import Dict, List, Tuple, Optional

from strategy_base import BaseStrategy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ── Instrument ID mapping (matches FFM training) ──
INSTRUMENT_IDS = {
    'ES': 0, 'MES': 0,
    'NQ': 1, 'MNQ': 1,
    'RTY': 2, 'MRTY': 2,
    'YM': 3, 'MYM': 3,
    'GC': 4, 'MGC': 4,
}

# ── Point values (for risk_dollars_norm feature) ──
POINT_VALUES = {
    'ES': 50.0,  'NQ': 20.0,  'RTY': 10.0,  'YM': 5.0,   'GC': 100.0,
    'MES': 5.0,  'MNQ': 2.0,  'MRTY': 5.0,  'MYM': 0.50, 'MGC': 10.0,
}

MAX_RISK_DOLLARS = 300.0   # normalization reference from training

# ── CISD detection params (must match training) ──
SWING_PERIOD       = 6
TOLERANCE          = 0.70
EXPIRY_BARS        = 50
LIQUIDITY_LOOKBACK = 10
ZONE_MAX_BARS      = 40
FIB_1              = 0.618
FIB_2              = 0.786
HTF_RANGE_BARS     = 96     # bars for P/D midpoint (~8hrs of 5min)
DISP_BODY_RATIO    = 0.50
DISP_CLOSE_STR     = 0.60

# ── Session windows ──
SESSION_START_HOUR = 7    # London open (NY time)
SESSION_END_HOUR   = 16   # RTH close
OPTIMAL_START_HOUR = 9    # in_optimal_session feature
OPTIMAL_END_HOUR   = 11


class CISDOTEStrategy(BaseStrategy):
    """
    CISD+OTE Hybrid Strategy using FFM Transformer backbone (v5.1).

    Architecture:
      FFMBackbone (256-dim, 42 features) + CISDProjection (28 features)
      → Fusion → SignalHead [noise_prob, signal_prob]
                 ConfidenceHead [0-1]
                 RiskHead [predicted R:R]

    Usage:
      strategy = CISDOTEStrategy(
          model_path='cisd_ote_hybrid_v5_1_F4.onnx',
          scaler_path='',           # no scaler needed
          contract_symbol='MNQ',
      )
    """

    def __init__(
        self,
        model_path: str,
        scaler_path: str,
        contract_symbol: str,
        session_start_hour: int = SESSION_START_HOUR,
        session_end_hour: int = SESSION_END_HOUR,
        min_vty_regime: float = 0.75,
    ):
        super().__init__(model_path, scaler_path, contract_symbol)
        self._session_start_hour = session_start_hour
        self._session_end_hour = session_end_hour

        # Regime gate — block trades when vty_regime (atr14/atr_ma50) is below threshold.
        # Stays persistently low during quiet periods — no rolling-window adaptation.
        # 0.0 = disabled, 0.8 = block when current ATR >20% below its 50-bar average.
        self._min_vty_regime = min_vty_regime
        self._latest_vty_regime: float = 1.0  # neutral until first bar computed

        # CISD zone tracker state
        self._active_zones: deque = deque(maxlen=20)
        self._last_bar_idx: int = 0

        # Pivot tracking state (rolling, no lookahead)
        self._pivot_highs: deque = deque(maxlen=200)  # (bar_idx, price)
        self._pivot_lows:  deque = deque(maxlen=200)
        self._pending_ph:  deque = deque(maxlen=50)   # bars waiting for confirmation
        self._pending_pl:  deque = deque(maxlen=50)
        self._last_wicked_high: int = -999
        self._last_wicked_low:  int = -999
        self._bear_pots:   deque = deque(maxlen=20)   # (open_price, bar_idx)
        self._bull_pots:   deque = deque(maxlen=20)

        # Latest signal state (set by add_features, read by predict)
        self._latest_cisd_features: Optional[np.ndarray] = None
        self._latest_zone_bullish: float = 0.0
        self._latest_risk_rr: float = 0.0

        # Bar counter for session/optimal tracking
        self._bar_count: int = 0

        logging.info("=" * 65)
        logging.info("🎯 CISD+OTE Strategy v5.1 — FFM Hybrid Transformer")
        logging.info("=" * 65)
        logging.info(f"  SEQ_LEN: 64 bars | CISD features: 28")
        logging.info(f"  Session: {SESSION_START_HOUR}:00–{SESSION_END_HOUR}:00 NY")
        logging.info(f"  Optimal window (feature): {OPTIMAL_START_HOUR}–{OPTIMAL_END_HOUR} ET")
        logging.info(f"  Confidence threshold recommended: 0.90 (conservative)")
        logging.info(f"  Warmup: ~200 bars for stable features")
        logging.info(f"  Direction: zone_is_bullish feature (CISD idx 4)")
        if min_vty_regime > 0.0:
            logging.info(f"  Regime gate: block when vty_regime < {min_vty_regime} (atr14/atr_ma50)")
        logging.info("=" * 65)

    # ── BaseStrategy interface ────────────────────────────────────

    def get_sequence_length(self) -> int:
        """FFM backbone uses 64-bar sequences."""
        return 64

    def get_feature_columns(self) -> List[str]:
        """
        42 FFM backbone feature columns.
        These must match get_model_feature_columns() from the FFM repo.
        The FFM repo computes these — we pull from the prepared parquet
        at training time. At inference we replicate the same calculations.
        """
        return [
            # Price/return features (10)
            'ret_1', 'ret_3', 'ret_5', 'ret_10', 'ret_20',
            'ret_acceleration', 'ret_momentum_10', 'ret_momentum_20',
            'ret_skew_20', 'ret_kurt_20',
            # Volatility features (8)
            'vty_atr_14', 'vty_atr_zscore', 'vty_range_ratio_20',
            'vty_atr_of_atr', 'vty_parkinson', 'vty_garman_klass',
            'vty_rolling_std_20', 'vty_regime',
            # Volume features (6)
            'vol_ratio_20', 'vol_ratio_60', 'vol_delta_proxy',
            'vol_imbalance', 'vol_vwap_dev', 'vol_acc',
            # Session/time features (8)
            'sess_time_of_day', 'sess_dist_from_open', 'sess_dist_from_vwap',
            'sess_dist_from_high', 'sess_dist_from_low',
            'sess_open_gap', 'sess_cum_ret', 'sess_bar_of_day',
            # Structure features (6)
            'str_structure_state', 'str_hh_count', 'str_ll_count',
            'str_swing_strength', 'str_trend_consistency', 'str_breakout_strength',
            # Market context (4)
            'ctx_day_of_week', 'ctx_week_of_year', 'ctx_month',
            'ctx_overnight_gap',
        ]

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute FFM backbone features + CISD+OTE zone detection.

        This runs on every bar close. The CISD detector is stateful —
        it updates incrementally using only the most recent bar, avoiding
        re-scanning the full history each time.

        Returns df with FFM feature columns added. Also updates
        self._latest_cisd_features for use in predict().
        """
        df = df.copy()
        n = len(df)

        if n < 2:
            self._latest_cisd_features = None
            return df

        # ── Compute FFM backbone features ──
        df = self._compute_ffm_features(df)

        # ── Update regime gate value ──
        if 'vty_regime' in df.columns:
            val = float(df['vty_regime'].iloc[-1])
            if np.isfinite(val):
                self._latest_vty_regime = val

        # ── Run incremental CISD detector on latest bar ──
        # On the first call with historical data, warm up the CISD state by
        # processing each prior bar individually so zone patterns from prefilled
        # history are visible (live mode prefills 200 bars without calling add_features).
        if self._bar_count == 0 and n > 1:
            logging.info(f"⏳ CISD warmup: processing {n - 1} historical bars...")
            for warmup_i in range(n - 1):
                self._update_cisd_detector(df.iloc[:warmup_i + 1], warmup_i)
                self._bar_count += 1
            logging.info(
                f"✅ CISD warmup done — {len(self._active_zones)} active zone(s), "
                f"{len(self._pivot_highs)} pivot highs, {len(self._pivot_lows)} pivot lows"
            )

        self._update_cisd_detector(df, self._bar_count)

        # ── Build 28-element CISD feature vector for latest bar ──
        self._latest_cisd_features = self._build_cisd_feature_vector(df, self._bar_count)

        self._bar_count += 1
        return df

    def load_model(self):
        """Load CISD+OTE ONNX model."""
        import os
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        self.model = onnxruntime.InferenceSession(
            self.model_path,
            providers=['CPUExecutionProvider'])
        # Log input/output names for debugging
        inputs  = [(i.name, i.shape) for i in self.model.get_inputs()]
        outputs = [(o.name, o.shape) for o in self.model.get_outputs()]
        logging.info(f"  ✅ ONNX loaded: {os.path.basename(self.model_path)}")
        logging.info(f"     Inputs:  {inputs}")
        logging.info(f"     Outputs: {outputs}")

    def load_scaler(self):
        """No scaler needed — FFM features are pre-normalized."""
        self.scaler = None
        logging.info("  ✅ No scaler required for CISD+OTE v5.1")

    def predict(self, df: pd.DataFrame) -> Tuple[int, float]:
        """
        Run ONNX inference and return (prediction, confidence).

        prediction:  0=Hold, 1=Buy, 2=Sell
        confidence:  signal_probs[0,1] — probability of signal class

        Direction mapping:
          signal_prob >= threshold AND zone_is_bullish > 0 → 1 (Buy)
          signal_prob >= threshold AND zone_is_bullish < 0 → 2 (Sell)
          signal_prob < threshold                          → 0 (Hold)
        """
        try:
            seq_len = self.get_sequence_length()
            feature_cols = self.get_feature_columns()

            if df.empty or len(df) < seq_len:
                logging.debug(f"⏳ Warmup: {len(df)}/{seq_len} bars")
                return 0, 0.0

            if self._latest_cisd_features is None:
                logging.debug("⏳ No active CISD zone — Hold")
                return 0, 0.0

            # Check required feature columns exist
            missing = [c for c in feature_cols if c not in df.columns]
            if missing:
                logging.warning(f"⚠️ Missing FFM features: {missing[:5]}...")
                return 0, 0.0

            # ── Build FFM sequence input [1, 64, 42] ──
            feat_arr = df[feature_cols].values.astype(np.float32)
            feat_arr = np.nan_to_num(feat_arr, nan=0.0, posinf=5.0, neginf=-5.0)
            feat_arr = np.clip(feat_arr, -10.0, 10.0)
            seq = feat_arr[-seq_len:].reshape(1, seq_len, -1)  # [1, 64, 42]

            # ── CISD features [1, 28] ──
            cisd = self._latest_cisd_features.reshape(1, -1).astype(np.float32)

            # ── Temporal inputs ──
            # time_of_day: normalized 0-1 position within 24h
            hours = df.index[-seq_len:].hour.values if hasattr(df.index, 'hour') \
                    else np.zeros(seq_len)
            mins  = df.index[-seq_len:].minute.values if hasattr(df.index, 'minute') \
                    else np.zeros(seq_len)
            tod   = ((hours * 60 + mins) / 1440.0).astype(np.float32)
            time_of_day = tod.reshape(1, seq_len)

            # day_of_week: 0-4
            dow = df.index[-seq_len:].dayofweek.values.astype(np.int64) \
                  if hasattr(df.index, 'dayofweek') else np.zeros(seq_len, dtype=np.int64)
            day_of_week = dow.reshape(1, seq_len)

            # instrument_id: scalar per batch
            base = self.contract_symbol.upper().split('.')[0][:4].rstrip('0123456789')
            inst_id = INSTRUMENT_IDS.get(base, 0)
            instrument_ids = np.array([inst_id], dtype=np.int64)

            # session_ids: 0=pre-market, 1=london, 2=ny, 3=after
            if hasattr(df.index, 'hour'):
                h = df.index[-seq_len:].hour
                sess = np.where(h < 7, 0,
                       np.where(h < 9, 1,
                       np.where(h < 16, 2, 3))).astype(np.int64)
            else:
                sess = np.full(seq_len, 2, dtype=np.int64)
            session_ids = sess.reshape(1, seq_len)

            # ── Run inference ──
            outputs = self.model.run(
                ['signal_probs', 'confidence', 'risk'],
                {
                    'features':       seq,
                    'cisd_features':  cisd,
                    'time_of_day':    time_of_day,
                    'day_of_week':    day_of_week,
                    'instrument_ids': instrument_ids,
                    'session_ids':    session_ids,
                }
            )

            signal_probs = outputs[0]   # [1, 2]
            confidence   = float(outputs[1][0])  # scalar
            self._latest_risk_rr = float(np.array(outputs[2]).flatten()[0])

            signal_prob = float(signal_probs[0, 1])

            # Map to prediction class using zone direction
            # (direction stored when zone was built — CISD feature idx 4)
            zone_bullish = self._latest_zone_bullish
            if zone_bullish > 0:
                prediction = 1   # BUY
            elif zone_bullish < 0:
                prediction = 2   # SELL
            else:
                prediction = 0   # No active zone

            logging.debug(
                f"  CISD+OTE | signal_prob={signal_prob:.3f} "
                f"conf={confidence:.3f} dir={'BUY' if prediction==1 else 'SELL' if prediction==2 else 'NONE'}"
            )

            # Return signal_prob as confidence (used by should_enter_trade threshold)
            return prediction, signal_prob

        except Exception as e:
            logging.exception(f"❌ CISD+OTE predict error: {e}")
            return 0, 0.0

    def is_trading_allowed(self, timestamp: pd.Timestamp) -> bool:
        """Block entries outside the configured session window (ET hours)."""
        hour = timestamp.hour
        allowed = self._session_start_hour <= hour < self._session_end_hour
        if not allowed:
            logging.debug(
                f"⏸ Entry blocked — {timestamp.strftime('%H:%M %Z')} outside "
                f"session [{self._session_start_hour:02d}:00–{self._session_end_hour:02d}:00]"
            )
        return allowed

    def should_enter_trade(
        self,
        prediction: int,
        confidence: float,
        bar: Dict,
        entry_conf: float,
        adx_thresh: float,
    ) -> Tuple[bool, Optional[str]]:
        """
        Gate entry on confidence threshold.

        Recommended thresholds (walk-forward v5.1):
          0.90 → 68.2% precision, PF 8.71  (conservative — prop firm)
          0.80 → 55.8% precision, PF 5.15  (moderate)
          0.70 → 51.7% precision, PF 4.22  (aggressive)

        adx_thresh is accepted for interface compatibility but ignored —
        the FFM model's confidence already encodes regime quality.
        """
        if confidence < entry_conf:
            return False, None

        # Regime gate — skip when market is in a sustained low-vol environment
        if self._min_vty_regime > 0.0 and self._latest_vty_regime < self._min_vty_regime:
            logging.info(
                f"🚫 Regime gate: vty_regime={self._latest_vty_regime:.3f} "
                f"< {self._min_vty_regime} — skipping"
            )
            return False, None

        if prediction == 1:
            logging.info(f"✅ CISD+OTE BUY  | signal_prob={confidence:.3f} vty_regime={self._latest_vty_regime:.3f}")
            return True, 'LONG'
        elif prediction == 2:
            logging.info(f"✅ CISD+OTE SELL | signal_prob={confidence:.3f} vty_regime={self._latest_vty_regime:.3f}")
            return True, 'SHORT'

        return False, None

    def on_trade_exit(self, reason: str):
        """Clear all active zones after a stop loss — zone is invalidated."""
        if reason == 'STOP_LOSS':
            self._active_zones.clear()
            self._latest_cisd_features = None
            self._latest_zone_bullish = 0.0
            logging.info("🚫 Stop loss — all CISD zones cleared")

    def get_stop_target_pts(self, df, direction, entry_price):
        """
        Return stop and target in points derived from the active OTE zone.

        Stop  = distance from entry to the far zone boundary
                (fib_bot for LONG, fib_top for SHORT).
        Target = stop_pts × predicted R:R from model's risk head (min 1.0).
        """
        if not self._active_zones:
            return None, None

        nearest = min(
            self._active_zones,
            key=lambda z: abs(entry_price - (z['fib_top'] + z['fib_bot']) / 2.0)
        )

        ft = nearest['fib_top']
        fb = nearest['fib_bot']

        if direction == 'LONG':
            stop_pts = abs(entry_price - fb)
        else:
            stop_pts = abs(ft - entry_price)

        if stop_pts <= 0:
            return None, None

        target_pts = stop_pts * 2.0

        logging.info(
            f"  CISD stop/target | dir={direction} entry={entry_price:.2f} "
            f"zone=[{fb:.2f}–{ft:.2f}] stop={stop_pts:.2f}pts "
            f"target={target_pts:.2f}pts (R:R 2.00)"
        )
        return stop_pts, target_pts

    # ── FFM Feature Computation ───────────────────────────────────

    def _compute_ffm_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the 42 FFM backbone features from OHLCV data.

        These replicate the derive_features() logic from the FFM repo.
        All features are normalized/clipped to match training scale.
        """
        c = df['close'].values.astype(np.float64)
        h = df['high'].values.astype(np.float64)
        l = df['low'].values.astype(np.float64)
        o = df['open'].values.astype(np.float64)
        v = df['volume'].values.astype(np.float64)
        n = len(df)
        eps = 1e-8

        # ── ATR (used for normalization) ──
        tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)),
                                           np.abs(l - np.roll(c, 1))))
        tr[0] = h[0] - l[0]
        atr14 = pd.Series(tr).ewm(span=14, adjust=False).mean().values
        atr_safe = np.where(atr14 > 0, atr14, eps)

        # ── Return features ──
        ret = np.diff(c, prepend=c[0]) / (np.roll(c, 1) + eps)
        ret[0] = 0.0

        df['ret_1']  = ret
        df['ret_3']  = pd.Series(c).pct_change(3).fillna(0).values
        df['ret_5']  = pd.Series(c).pct_change(5).fillna(0).values
        df['ret_10'] = pd.Series(c).pct_change(10).fillna(0).values
        df['ret_20'] = pd.Series(c).pct_change(20).fillna(0).values

        ret_s = pd.Series(ret)
        df['ret_acceleration']  = ret_s.diff(5).fillna(0).values * 100
        df['ret_momentum_10']   = ret_s.rolling(10).sum().fillna(0).values * 100
        df['ret_momentum_20']   = ret_s.rolling(20).sum().fillna(0).values * 100
        df['ret_skew_20']       = ret_s.rolling(20).skew().fillna(0).values
        df['ret_kurt_20']       = ret_s.rolling(20).kurt().fillna(0).values

        # ── Volatility features ──
        df['vty_atr_14']       = atr14 / (c + eps)
        atr_z_mean = pd.Series(atr14).rolling(100).mean().fillna(method='bfill').values
        atr_z_std  = pd.Series(atr14).rolling(100).std().fillna(1.0).values
        df['vty_atr_zscore']   = (atr14 - atr_z_mean) / (atr_z_std + eps)
        range20 = pd.Series(h - l).rolling(20).mean().fillna(h[0] - l[0]).values
        df['vty_range_ratio_20'] = (h - l) / (range20 + eps)
        df['vty_atr_of_atr']   = pd.Series(atr14).rolling(14).std().fillna(0).values / (atr14 + eps)
        # Parkinson estimator
        df['vty_parkinson']    = np.sqrt(np.maximum(
            pd.Series(np.log(h / (l + eps)) ** 2).rolling(20).mean().fillna(0).values / (4 * np.log(2)), 0))
        # Garman-Klass
        gk = 0.5 * np.log((h + eps) / (l + eps))**2 - (2*np.log(2)-1) * np.log((c + eps) / (o + eps))**2
        df['vty_garman_klass'] = pd.Series(np.sqrt(np.maximum(gk, 0))).rolling(20).mean().fillna(0).values
        df['vty_rolling_std_20'] = pd.Series(ret).rolling(20).std().fillna(0).values
        atr_ma50 = pd.Series(atr14).rolling(50).mean().fillna(method='bfill').values
        df['vty_regime']       = atr14 / (atr_ma50 + eps)

        # ── Volume features ──
        vol_ma20 = pd.Series(v).rolling(20).mean().fillna(method='bfill').values
        vol_ma60 = pd.Series(v).rolling(60).mean().fillna(method='bfill').values
        df['vol_ratio_20']     = v / (vol_ma20 + eps)
        df['vol_ratio_60']     = v / (vol_ma60 + eps)
        bar_delta              = np.where(c > o, v, np.where(c < o, -v, 0.0))
        df['vol_delta_proxy']  = bar_delta / (v + eps)

        # Buy/sell imbalance proxy
        buy_vol  = np.where(c > o, v, 0.0)
        sell_vol = np.where(c < o, v, 0.0)
        df['vol_imbalance']    = (buy_vol - sell_vol) / (v + eps)

        # VWAP deviation (session-based)
        dates = df.index.date if hasattr(df.index, 'date') else np.zeros(n)
        vwap_dev = np.zeros(n)
        if hasattr(df.index, 'date'):
            tmp = pd.DataFrame({'pv': c * v, 'v': v, 'date': df.index.date})
            cum_pv = tmp.groupby('date')['pv'].cumsum().values
            cum_v  = tmp.groupby('date')['v'].cumsum().values
            vwap   = cum_pv / (cum_v + eps)
            vwap_dev = (c - vwap) / atr_safe
        df['vol_vwap_dev']     = vwap_dev

        # Volume acceleration
        df['vol_acc']          = pd.Series(v).pct_change(5).fillna(0).values

        # ── Session features ──
        if hasattr(df.index, 'hour'):
            hours = df.index.hour
            mins  = df.index.minute
            df['sess_time_of_day'] = (hours * 60 + mins) / 1440.0
        else:
            df['sess_time_of_day'] = 0.5

        # Distance from session open
        sess_open = np.zeros(n)
        if hasattr(df.index, 'date'):
            tmp2 = pd.DataFrame({'c': c, 'date': df.index.date})
            open_map = tmp2.groupby('date')['c'].first().to_dict()
            sess_open = np.array([open_map.get(d, c[i]) for i, d in enumerate(df.index.date)])
        df['sess_dist_from_open'] = (c - sess_open) / (atr_safe)

        df['sess_dist_from_vwap'] = vwap_dev  # reuse above
        # Rolling session high/low
        sess_high = pd.Series(h).rolling(78, min_periods=1).max().values  # ~6.5hr session
        sess_low  = pd.Series(l).rolling(78, min_periods=1).min().values
        df['sess_dist_from_high'] = (c - sess_high) / (atr_safe)
        df['sess_dist_from_low']  = (c - sess_low)  / (atr_safe)

        prev_c = np.roll(c, 78); prev_c[:78] = c[0]
        df['sess_open_gap']       = (o - prev_c) / (atr_safe)

        df['sess_cum_ret'] = pd.Series(ret).rolling(78).sum().fillna(0).values
        df['sess_bar_of_day'] = np.arange(n) % 78 / 78.0

        # ── Structure features ──
        # HH/LL state using 20-bar rolling
        roll_high = pd.Series(h).rolling(20).max().values
        roll_low  = pd.Series(l).rolling(20).min().values
        prev_roll_h = np.roll(roll_high, 20); prev_roll_h[:20] = roll_high[0]
        prev_roll_l = np.roll(roll_low,  20); prev_roll_l[:20] = roll_low[0]
        hh = (roll_high > prev_roll_h)
        hl = (roll_low  > prev_roll_l)
        ll = (roll_low  < prev_roll_l)
        lh = (roll_high < prev_roll_h)
        structure = np.where(hh & hl, 1, np.where(ll & lh, -1, 0)).astype(float)
        df['str_structure_state'] = structure

        df['str_hh_count'] = pd.Series(hh).rolling(20).sum().fillna(0).values / 20
        df['str_ll_count'] = pd.Series(ll).rolling(20).sum().fillna(0).values / 20

        swing_range = roll_high - roll_low
        df['str_swing_strength'] = swing_range / (atr_safe * 20)

        df['str_trend_consistency'] = pd.Series(structure).rolling(20).mean().fillna(0).values
        breakout = np.where(c > roll_high, (c - roll_high) / atr_safe,
                   np.where(c < roll_low,  (roll_low - c) / atr_safe, 0.0))
        df['str_breakout_strength'] = breakout

        # ── Context features ──
        if hasattr(df.index, 'dayofweek'):
            df['ctx_day_of_week']  = df.index.dayofweek.values / 4.0
            df['ctx_week_of_year'] = df.index.isocalendar().week.values.astype(float) / 52.0
            df['ctx_month']        = (df.index.month.values - 1) / 11.0
        else:
            df['ctx_day_of_week']  = 0.0
            df['ctx_week_of_year'] = 0.0
            df['ctx_month']        = 0.0

        prev_open = np.roll(o, 78); prev_open[:78] = o[0]
        prev_close_bar = np.roll(c, 78); prev_close_bar[:78] = c[0]
        df['ctx_overnight_gap'] = (o - prev_close_bar) / (atr_safe)

        # ── Final clip / NaN fill ──
        feature_cols = self.get_feature_columns()
        for col in feature_cols:
            if col in df.columns:
                df[col] = pd.Series(df[col]).replace([np.inf, -np.inf], np.nan) \
                                            .fillna(method='ffill').fillna(0).values
                df[col] = np.clip(df[col].values, -10.0, 10.0)
            else:
                df[col] = 0.0

        return df

    # ── Incremental CISD Detector ─────────────────────────────────

    def _update_cisd_detector(self, df: pd.DataFrame, abs_bar: int):
        """
        Incremental CISD detector using absolute bar indices for all stored state.
        Pivots, pots, and zones store absolute bar numbers so state stays valid
        across calls where df is a sliding window (rel index is always 0..n-1).
        """
        n   = len(df)
        bar = n - 1   # df-relative index of current bar (always n-1)

        if bar < 1:
            return

        o_arr = df['open'].values
        h_arr = df['high'].values
        l_arr = df['low'].values
        c_arr = df['close'].values

        def to_rel(abs_idx):
            """Convert absolute bar index to df-relative, clamped to window start."""
            return max(0, n - 1 - abs_bar + abs_idx)

        # ── Track sweep (wick-throughs) ──
        new_ph = []
        for ph_price, ph_abs in self._pivot_highs:
            if (abs_bar - ph_abs) >= EXPIRY_BARS:
                continue
            if h_arr[bar] >= ph_price:
                self._last_wicked_high = abs_bar
            else:
                new_ph.append((ph_price, ph_abs))
        self._pivot_highs = deque(new_ph, maxlen=200)

        new_pl = []
        for pl_price, pl_abs in self._pivot_lows:
            if (abs_bar - pl_abs) >= EXPIRY_BARS:
                continue
            if l_arr[bar] <= pl_price:
                self._last_wicked_low = abs_bar
            else:
                new_pl.append((pl_price, pl_abs))
        self._pivot_lows = deque(new_pl, maxlen=200)

        # ── Pivot confirmation ──
        confirm_rel = bar - SWING_PERIOD
        confirm_abs = abs_bar - SWING_PERIOD
        if confirm_rel >= 1:
            window = h_arr[max(0, confirm_rel - SWING_PERIOD):
                           min(n, confirm_rel + SWING_PERIOD + 1)]
            if len(window) == 2 * SWING_PERIOD + 1:
                center = h_arr[confirm_rel]
                if center == window.max() and (window == center).sum() == 1:
                    self._pivot_highs.append((center, confirm_abs))

            window = l_arr[max(0, confirm_rel - SWING_PERIOD):
                           min(n, confirm_rel + SWING_PERIOD + 1)]
            if len(window) == 2 * SWING_PERIOD + 1:
                center = l_arr[confirm_rel]
                if center == window.min() and (window == center).sum() == 1:
                    self._pivot_lows.append((center, confirm_abs))

        # ── Track potential CISD candles (stored with absolute indices) ──
        if c_arr[bar-1] < o_arr[bar-1] and c_arr[bar] > o_arr[bar]:
            self._bear_pots.append((o_arr[bar], abs_bar))
        if c_arr[bar-1] > o_arr[bar-1] and c_arr[bar] < o_arr[bar]:
            self._bull_pots.append((o_arr[bar], abs_bar))

        self._bear_pots = deque(
            [(p, b) for p, b in self._bear_pots if abs_bar - b < EXPIRY_BARS],
            maxlen=20)
        self._bull_pots = deque(
            [(p, b) for p, b in self._bull_pots if abs_bar - b < EXPIRY_BARS],
            maxlen=20)

        # ── P/D midpoint ──
        rng_h = h_arr[max(0, bar - HTF_RANGE_BARS):bar].max() if bar > 0 else h_arr[bar]
        rng_l = l_arr[max(0, bar - HTF_RANGE_BARS):bar].min() if bar > 0 else l_arr[bar]
        pd_mid = (rng_h + rng_l) / 2.0

        # ── Bearish CISD check ──
        cisd_zone = None
        cisd_dir  = None
        new_bear  = deque(maxlen=20)
        for pot_price, pot_abs in self._bear_pots:
            pot_rel = to_rel(pot_abs)
            if c_arr[bar] < pot_price:
                highest_c = c_arr[pot_rel:bar+1].max()
                top_level = 0.0
                idx = pot_rel + 1
                while idx < bar and c_arr[idx] < o_arr[idx]:
                    top_level = o_arr[idx]; idx += 1
                if top_level > 0 and (top_level - pot_price) > 0:
                    ratio = (highest_c - pot_price) / (top_level - pot_price)
                    if ratio > TOLERANCE:
                        full_range = h_arr[bar] - l_arr[bar]
                        body = abs(c_arr[bar] - o_arr[bar])
                        br = body / full_range if full_range > 0 else 0.0
                        cs = (h_arr[bar] - c_arr[bar]) / full_range if full_range > 0 else 0.0
                        if br >= DISP_BODY_RATIO and cs >= DISP_CLOSE_STR:
                            in_premium = c_arr[bar] > pd_mid
                            had_sweep  = (abs_bar - self._last_wicked_high) <= LIQUIDITY_LOOKBACK
                            if in_premium or had_sweep:
                                h_max = h_arr[pot_rel:bar+1].max()
                                diff  = h_max - l_arr[bar]
                                ft = h_max - diff * FIB_1
                                fb = h_max - diff * FIB_2
                                fib_top = max(ft, fb); fib_bot = min(ft, fb)
                                if fib_top > fib_bot:
                                    cisd_zone = {
                                        'is_bullish':    False,
                                        'fib_top':       fib_top,
                                        'fib_bot':       fib_bot,
                                        'created_bar':   abs_bar,
                                        'had_sweep':     had_sweep,
                                        'disp_strength': float(ratio),
                                        'signal_fired':  False,
                                        'entered_zone':  False,
                                    }
                                    cisd_dir = 'BEAR'
                            break
            else:
                new_bear.append((pot_price, pot_abs))
        if cisd_dir is None:
            self._bear_pots = new_bear

        # ── Bullish CISD check ──
        new_bull = deque(maxlen=20)
        for pot_price, pot_abs in self._bull_pots:
            pot_rel = to_rel(pot_abs)
            if c_arr[bar] > pot_price and cisd_dir is None:
                lowest_c     = c_arr[pot_rel:bar+1].min()
                bottom_level = 0.0
                idx = pot_rel + 1
                while idx < bar and c_arr[idx] > o_arr[idx]:
                    bottom_level = o_arr[idx]; idx += 1
                if bottom_level > 0 and (pot_price - bottom_level) > 0:
                    ratio = (pot_price - lowest_c) / (pot_price - bottom_level)
                    if ratio > TOLERANCE:
                        full_range = h_arr[bar] - l_arr[bar]
                        body = abs(c_arr[bar] - o_arr[bar])
                        br = body / full_range if full_range > 0 else 0.0
                        cs = (c_arr[bar] - l_arr[bar]) / full_range if full_range > 0 else 0.0
                        if br >= DISP_BODY_RATIO and cs >= DISP_CLOSE_STR:
                            in_discount = c_arr[bar] <= pd_mid
                            had_sweep   = (abs_bar - self._last_wicked_low) <= LIQUIDITY_LOOKBACK
                            if in_discount or had_sweep:
                                l_min = l_arr[pot_rel:bar+1].min()
                                diff  = h_arr[bar] - l_min
                                ft = l_min + diff * FIB_1
                                fb = l_min + diff * FIB_2
                                fib_top = max(ft, fb); fib_bot = min(ft, fb)
                                if fib_top > fib_bot:
                                    cisd_zone = {
                                        'is_bullish':    True,
                                        'fib_top':       fib_top,
                                        'fib_bot':       fib_bot,
                                        'created_bar':   abs_bar,
                                        'had_sweep':     had_sweep,
                                        'disp_strength': float(ratio),
                                        'signal_fired':  False,
                                        'entered_zone':  False,
                                    }
                                    cisd_dir = 'BULL'
                            break
            else:
                new_bull.append((pot_price, pot_abs))
        if cisd_dir is None:
            self._bull_pots = new_bull

        if cisd_zone is not None:
            self._active_zones.appendleft(cisd_zone)

        # ── Update active zones ──
        current_close = c_arr[bar]
        current_high  = h_arr[bar]
        current_low   = l_arr[bar]
        current_open  = o_arr[bar]

        surviving = []
        for z in self._active_zones:
            if abs_bar - z['created_bar'] > ZONE_MAX_BARS:
                continue
            if z['is_bullish'] and current_close < z['fib_bot']:
                continue
            if not z['is_bullish'] and current_close > z['fib_top']:
                continue
            if current_low <= z['fib_top'] and current_high >= z['fib_bot']:
                z['entered_zone'] = True
            surviving.append(z)

        self._active_zones = deque(surviving, maxlen=20)

        # ── Check for OTE entry ──
        for z in self._active_zones:
            if z.get('signal_fired'):
                continue
            if not z['entered_zone']:
                continue
            if abs_bar <= z['created_bar']:
                continue
            touched = current_low <= z['fib_top'] and current_high >= z['fib_bot']
            if not touched:
                continue
            confirmed = current_close > current_open if z['is_bullish'] \
                        else current_close < current_open
            if confirmed:
                z['signal_fired'] = True
                z['entry_bar']    = abs_bar
                z['entry_price']  = current_close
                logging.debug(
                    f"  OTE zone touched | {'BULL' if z['is_bullish'] else 'BEAR'} "
                    f"zone [{z['fib_bot']:.2f}–{z['fib_top']:.2f}] "
                    f"entry @ {current_close:.2f}"
                )
                break

    def _build_cisd_feature_vector(self, df: pd.DataFrame, abs_bar: int) -> Optional[np.ndarray]:
        """
        Build the 28-element CISD feature vector for the latest bar.
        Returns None if no active zone exists.

        Feature order must exactly match CISD_FEATURE_COLS from training:
          0:  zone_height_vs_atr
          1:  price_vs_zone_top
          2:  price_vs_zone_bot
          3:  zone_age_bars
          4:  zone_is_bullish          ← direction flag (+1=BUY, -1=SELL)
          5:  cisd_displacement_strength
          6:  had_liquidity_sweep
          7:  htf_trend_direction
          8:  trend_alignment
          9:  rejection_wick_ratio
          10: close_position
          11: volume_trend
          12: cumulative_delta_ratio
          13: price_vs_ema20
          14: gap_from_prior_close
          15: session_progress
          16: day_of_week_feat
          17: confluence_score
          18: risk_dollars_norm
          19: in_optimal_session
          20: entry_distance_pct
          21: ffm_sess_dist_from_vwap
          22: ffm_str_structure_state
          23: ffm_ret_acceleration
          24: ffm_vty_atr_of_atr
          25: ffm_sess_dist_from_open
          26: ffm_ret_momentum_10
          27: ffm_vol_delta_proxy
        """
        if not self._active_zones:
            self._latest_zone_bullish = 0.0
            return None

        # Find nearest active zone (by midpoint distance to current close)
        bar = len(df) - 1
        c   = float(df['close'].iloc[-1])
        h   = float(df['high'].iloc[-1])
        l   = float(df['low'].iloc[-1])
        o   = float(df['open'].iloc[-1])
        v   = float(df['volume'].iloc[-1])

        # ATR for normalization
        atr_col = 'vty_atr_14'
        if atr_col in df.columns:
            atr_raw = float(df[atr_col].iloc[-1]) * c  # un-normalize: atr/c * c
        else:
            atr_raw = float((h - l) * 14)  # rough fallback
        atr_safe = max(atr_raw, 1e-6)

        nearest = None
        nearest_dist = float('inf')
        for z in self._active_zones:
            # Skip zones that already fired on a previous bar (consumed)
            if z.get('signal_fired') and z.get('entry_bar', abs_bar) < abs_bar:
                continue
            mid = (z['fib_top'] + z['fib_bot']) / 2.0
            d   = abs(c - mid)
            if d < nearest_dist:
                nearest_dist = d; nearest = z

        if nearest is None:
            self._latest_zone_bullish = 0.0
            return None

        z = nearest
        self._latest_zone_bullish = 1.0 if z['is_bullish'] else -1.0

        ft = z['fib_top']; fb = z['fib_bot']
        zh = ft - fb; zh_safe = max(zh, 1e-6)
        age = float(abs_bar - z['created_bar'])

        # HTF trend from structure feature
        htf_trend = float(df['str_structure_state'].iloc[-1]) \
                    if 'str_structure_state' in df.columns else 0.0
        is_aligned = float(
            (z['is_bullish'] and htf_trend >= 0) or
            (not z['is_bullish'] and htf_trend <= 0))

        # Candle quality
        cr       = max(h - l, 1e-6)
        body     = abs(c - o)
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        wick     = lower_wick if z['is_bullish'] else upper_wick
        body_safe = max(body, 1e-6)
        rej_wick = wick / body_safe
        close_pos = (c - l) / cr

        # Volume trend
        if 'vol_ratio_20' in df.columns:
            vol_trend = float(df['vol_ratio_20'].iloc[-1])
        else:
            vol_trend = 1.0

        # Cumulative delta ratio
        if 'vol_delta_proxy' in df.columns:
            cum_delta = float(df['vol_delta_proxy'].iloc[-1])
        else:
            cum_delta = 0.0

        # Price vs EMA20
        if 'ret_momentum_20' in df.columns:
            price_vs_ema20 = float(df['ret_momentum_20'].iloc[-1])
        else:
            price_vs_ema20 = 0.0

        # Gap from prior close
        if 'ctx_overnight_gap' in df.columns:
            gap = float(df['ctx_overnight_gap'].iloc[-1])
        else:
            gap = 0.0

        # Session progress (0-1 within 7-16 NY)
        if hasattr(df.index, 'hour'):
            hh = df.index[-1].hour
            mm = df.index[-1].minute
            sess_mins = max(hh * 60 + mm - SESSION_START_HOUR * 60, 0)
            sess_prog = min(sess_mins / 540.0, 1.0)  # 9h window = 540 min

            in_optimal = 1.0 if (OPTIMAL_START_HOUR <= hh < OPTIMAL_END_HOUR) else 0.0
        else:
            sess_prog  = 0.5
            in_optimal = 0.0

        # Day of week
        dow = float(df.index[-1].dayofweek) if hasattr(df.index[-1], 'dayofweek') else 0.0

        # Confluence score (0-10)
        score = 2.0
        if z['had_sweep']:      score += 2.0
        if is_aligned:          score += 1.0
        if z['disp_strength'] >= 0.6: score += 1.0
        if in_optimal:          score += 1.0

        # Risk dollars norm
        base = self.contract_symbol.upper().split('.')[0][:4].rstrip('0123456789')
        pv   = POINT_VALUES.get(base, 20.0)
        sl   = fb if z['is_bullish'] else ft
        risk_pts = abs(c - sl)
        risk_norm = min((risk_pts * pv) / MAX_RISK_DOLLARS, 5.0)

        # Entry distance
        entry_dist = abs(c - (ft + fb) / 2.0) / zh_safe

        # FFM-derived features (pull from computed columns)
        def ffm(col, default=0.0):
            return float(df[col].iloc[-1]) if col in df.columns else default

        features = np.array([
            # Zone context (5)
            np.clip(zh / atr_safe, 0, 10),              # 0: zone_height_vs_atr
            np.clip((c - ft) / zh_safe, -2, 5),          # 1: price_vs_zone_top
            np.clip((c - fb) / zh_safe, -2, 5),          # 2: price_vs_zone_bot
            np.clip(age / ZONE_MAX_BARS, 0, 5),           # 3: zone_age_bars
            self._latest_zone_bullish,                    # 4: zone_is_bullish ← KEY
            # Displacement (4)
            np.clip(z['disp_strength'], 0, 5),            # 5: cisd_displacement_strength
            1.0 if z['had_sweep'] else 0.0,               # 6: had_liquidity_sweep
            np.clip(htf_trend, -1, 1),                    # 7: htf_trend_direction
            is_aligned,                                   # 8: trend_alignment
            # Candle quality (2)
            np.clip(rej_wick, 0, 10),                     # 9: rejection_wick_ratio
            np.clip(close_pos, 0, 1),                     # 10: close_position
            # Volume (2)
            np.clip(vol_trend, 0, 5),                     # 11: volume_trend
            np.clip(cum_delta, -1, 1),                    # 12: cumulative_delta_ratio
            # Market context (2)
            np.clip(price_vs_ema20, -10, 10),             # 13: price_vs_ema20
            np.clip(gap, -10, 10),                        # 14: gap_from_prior_close
            # Timing (2)
            np.clip(sess_prog, 0, 1),                     # 15: session_progress
            dow / 4.0,                                    # 16: day_of_week_feat
            # Filter states (4)
            np.clip(score / 10.0, 0, 1),                  # 17: confluence_score
            np.clip(risk_norm, 0, 5),                     # 18: risk_dollars_norm
            in_optimal,                                   # 19: in_optimal_session
            np.clip(entry_dist, -2, 5),                   # 20: entry_distance_pct
            # FFM-pulled features (7)
            np.clip(ffm('sess_dist_from_vwap'), -10, 10), # 21: ffm_sess_dist_from_vwap
            np.clip(ffm('str_structure_state'), -5, 5),   # 22: ffm_str_structure_state
            np.clip(ffm('ret_acceleration'), -10, 10),       # 23: ffm_ret_acceleration
            np.clip(ffm('vty_atr_of_atr'), -5, 5),        # 24: ffm_vty_atr_of_atr
            np.clip(ffm('sess_dist_from_open'), -10, 10), # 25: ffm_sess_dist_from_open
            np.clip(ffm('ret_momentum_10'), -10, 10),      # 26: ffm_ret_momentum_10
            np.clip(ffm('vol_delta_proxy'), -1, 1),        # 27: ffm_vol_delta_proxy
        ], dtype=np.float32)

        features = np.nan_to_num(features, nan=0.0, posinf=5.0, neginf=-5.0)
        features = np.clip(features, -10.0, 10.0)
        return features

    def preprocess_features(self, df: pd.DataFrame) -> np.ndarray:
        """Override base: no scaler, just return normalized array."""
        feature_cols = self.get_feature_columns()
        arr = df[feature_cols].values.astype(np.float32)
        return np.nan_to_num(arr, nan=0.0, posinf=5.0, neginf=-5.0)