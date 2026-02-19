#!/usr/bin/env python3
"""
ORB Hybrid Strategy V1.0 - PRODUCTION-READY

Opening Range Breakout strategy powered by the FFM (Futures Foundation Model)
hybrid architecture. Multi-instrument, multi-session AI signal generation.

Architecture:
  - FFMBackbone: 64-bar sequence of market microstructure features
  - ORB projection: 20 point-in-time opening range features
  - Fusion + signal head + confidence head
  - Exported as ONNX for fast inference

Walk-Forward OOS Results (3.5 years):
  ≥0.90 confidence: PF 2.19, +31R, 31.6% win rate
  ≥0.95 confidence: PF 3.25, +36R, 40.7% win rate
  Avg R:R ~4.8R

DATA REQUIREMENTS:
  Minimum : 200 bars (~10 hours) for ORB feature warmup
  Recommended: 500+ bars for stable ATR and volume baselines

SESSIONS SUPPORTED:
  Asia   : 18:00 – 01:00 ET  (ORB period: 18:00 – 19:00)
  London : 03:00 – 08:30 ET  (ORB period: 03:00 – 04:00)
  NY     : 09:30 – 16:00 ET  (ORB period: 09:30 – 10:00)
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import onnxruntime
from typing import Dict, List, Tuple, Optional
from strategy_base import BaseStrategy

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')


# ══════════════════════════════════════════════════════════════
#   SESSION DEFINITIONS
# ══════════════════════════════════════════════════════════════

SESSIONS = {
    'Asia': {
        'start_hour': 18, 'start_min': 0,
        'end_hour':    1,  'end_min':   0,
        'orb_end_hour': 19, 'orb_end_min': 0,
        'session_id':  0,
    },
    'London': {
        'start_hour': 3,  'start_min': 0,
        'end_hour':   8,  'end_min':   30,
        'orb_end_hour': 4, 'orb_end_min': 0,
        'session_id': 1,
    },
    'NY': {
        'start_hour': 9,  'start_min': 30,
        'end_hour':   16, 'end_min':   0,
        'orb_end_hour': 10, 'orb_end_min': 0,
        'session_id': 2,
    },
}

INSTRUMENT_IDS = {
    'ES': 0, 'NQ': 1, 'RTY': 2, 'YM': 3, 'GC': 4,
    'CL': 5, 'SI': 6, 'ZN':  7, 'ZB': 8,
}


class ORBHybridStrategy(BaseStrategy):
    """
    ORB Hybrid V1.0 — Production Live Trading Strategy

    Extends BaseStrategy with:
      - Session-aware ORB range tracking
      - 20 ORB-specific features matching training pipeline exactly
      - FFM sequence features (64-bar context window)
      - Multi-input ONNX inference
      - Per-session ORB state management
    """

    def __init__(
        self,
        model_path:      str,
        metadata_path:   str,
        contract_symbol: str,
        confidence_threshold: float = 0.90,
        bar_minutes:     int  = 5,
        orb_period_min:  int  = 60,
    ):
        """
        Args:
            model_path:           Path to .onnx file exported by Cell 5
            metadata_path:        Path to _metadata.json saved alongside .onnx
            contract_symbol:      e.g. 'ES', 'NQ', 'RTY', 'YM', 'GC'
            confidence_threshold: Minimum confidence to emit a signal (default 0.90)
            bar_minutes:          Bar size in minutes (must match training, default 5)
            orb_period_min:       Opening range period in minutes (default 60)
        """
        # BaseStrategy expects scaler_path — pass None since FFM doesn't use one
        super().__init__(model_path, scaler_path=None,
                         contract_symbol=contract_symbol)

        self.metadata_path        = metadata_path
        self.confidence_threshold = confidence_threshold
        self.bar_minutes          = bar_minutes
        self.orb_period_min       = orb_period_min
        self.metadata             = None
        self.bar_count            = 0
        self.last_quality         = None

        # Per-session ORB state — reset each new session
        self._orb_state: Dict[str, Dict] = {
            sess: self._empty_orb_state() for sess in SESSIONS
        }

        # Determine instrument id from symbol
        base = contract_symbol.split('.')[0][:2]
        self.instrument_id = INSTRUMENT_IDS.get(base, 0)
        logging.info(f"ORBHybridStrategy init: {contract_symbol} "
                     f"(instrument_id={self.instrument_id})")

    # ══════════════════════════════════════════════════════════
    #   ABSTRACT METHOD IMPLEMENTATIONS
    # ══════════════════════════════════════════════════════════

    def get_feature_columns(self) -> List[str]:
        """
        Returns FFM backbone feature columns loaded from metadata JSON.
        Falls back to a sensible default if metadata not loaded yet.
        """
        if self.metadata and 'feature_cols' in self.metadata:
            return self.metadata['feature_cols']
        # Minimal fallback — will be overridden once metadata loads
        return []

    def get_orb_feature_columns(self) -> List[str]:
        """Returns the 20 ORB-specific feature column names."""
        if self.metadata and 'orb_feature_cols' in self.metadata:
            return self.metadata['orb_feature_cols']
        return self._default_orb_feature_columns()

    def get_sequence_length(self) -> int:
        """64-bar context window matching training."""
        if self.metadata:
            return self.metadata.get('seq_len', 64)
        return 64

    def load_model(self):
        """Load ONNX model and metadata."""
        # Load metadata first
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(
                f"Metadata not found: {self.metadata_path}\n"
                f"Export model with Cell 5 to generate this file.")
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)
        logging.info(f"✅ Loaded metadata: fold={self.metadata.get('fold')}, "
                     f"checkpoint={self.metadata.get('checkpoint_key')}")

        # Load ONNX session
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.model = onnxruntime.InferenceSession(
            self.model_path, providers=providers)
        logging.info(f"✅ Loaded ONNX model: {os.path.basename(self.model_path)}")

    def load_scaler(self):
        """No external scaler — FFM normalizes internally."""
        self.scaler = None
        logging.info("ℹ️  No scaler needed — FFM normalizes inputs internally")

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all features required for ORB Hybrid inference.

        Two feature sets are produced:
          1. FFM features  — market microstructure (from futures_foundation pipeline)
          2. ORB features  — opening range statistics (20 point-in-time values)

        Both sets are stored as columns in the returned DataFrame so that
        predict() can slice them out for the ONNX model.

        Args:
            df: DataFrame with columns: open, high, low, close, volume
                Must have a DatetimeIndex in Eastern Time (or UTC+offset).

        Returns:
            df with all feature columns added.
        """
        self.bar_count += 1
        df = df.copy()

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex")

        # Convert to ET if timezone-aware
        if df.index.tz is not None:
            df.index = df.index.tz_convert('America/New_York')
        else:
            df.index = df.index.tz_localize('America/New_York',
                                             ambiguous='infer',
                                             nonexistent='shift_forward')

        # ── Quality gate ──
        quality = self._get_quality_level(len(df))
        if quality != self.last_quality:
            self.last_quality = quality
            logging.info(f"Data quality: {quality} ({len(df)} bars)")

        # ── Base price features (replicate FFM pipeline) ──
        df = self._add_ffm_features(df)

        # ── Session classification ──
        df = self._add_session_features(df)

        # ── ORB tracking and features ──
        df = self._add_orb_features(df)

        # ── Instrument id column ──
        df['_instrument_id'] = self.instrument_id

        # ── Clean up ──
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.ffill(inplace=True)
        df.fillna(0, inplace=True)

        return df

    def predict(self, df: pd.DataFrame) -> Tuple[int, float]:
        """
        Run ORB Hybrid inference on the latest bar.

        Returns:
            (prediction, confidence)
            prediction: 0=HOLD, 1=BUY, 2=SELL
            confidence: 0-1 model confidence score
        """
        try:
            seq_len = self.get_sequence_length()

            # Minimum data check
            if len(df) < max(seq_len, 200):
                if self.bar_count % 50 == 0:
                    logging.info(f"⏳ Accumulating: {len(df)}/{max(seq_len,200)} bars")
                return 0, 0.0

            # Check ORB is valid for current session
            session = self._get_current_session(df.index[-1])
            if session is None:
                return 0, 0.0
            orb = self._orb_state[session]
            if not orb['valid']:
                return 0, 0.0

            # ── Build inputs ──
            feature_cols = self.get_feature_columns()
            orb_cols     = self.get_orb_feature_columns()

            # Validate columns exist
            missing_ffm = [c for c in feature_cols if c not in df.columns]
            missing_orb = [c for c in orb_cols     if c not in df.columns]
            if missing_ffm or missing_orb:
                logging.warning(f"Missing features — FFM:{missing_ffm} ORB:{missing_orb}")
                return 0, 0.0

            # Sequence inputs — last seq_len bars
            seq_data = df[feature_cols].values[-seq_len:].astype(np.float32)
            if seq_data.shape[0] < seq_len:
                return 0, 0.0

            # ORB point-in-time input — last bar only
            orb_data = df[orb_cols].values[-1].astype(np.float32)

            # Temporal inputs
            time_of_day  = df['sess_time_of_day'].values[-seq_len:].astype(np.float32)
            day_of_week  = df['tmp_day_of_week'].values[-seq_len:].astype(np.int64)
            session_ids  = df['sess_id'].values[-seq_len:].astype(np.int64)

            # Sanity check for NaN/Inf
            for name, arr in [('features', seq_data), ('orb', orb_data)]:
                if np.isnan(arr).any() or np.isinf(arr).any():
                    logging.warning(f"Invalid values in {name} — skipping bar")
                    return 0, 0.0

            # ── ONNX inference ──
            ort_inputs = {
                'features':       seq_data[None],           # [1, 64, F]
                'orb_features':   orb_data[None],           # [1, 20]
                'time_of_day':    time_of_day[None],        # [1, 64]
                'day_of_week':    day_of_week[None],        # [1, 64]
                'instrument_ids': np.array([self.instrument_id], dtype=np.int64),
                'session_ids':    session_ids[None],        # [1, 64]
            }

            signal_probs, confidence, risk = self.model.run(None, ort_inputs)

            # Unpack
            probs      = signal_probs[0]   # [3]
            conf       = float(confidence[0])
            risk_rr    = float(risk[0, 0]) if risk.ndim == 2 else float(risk[0])
            prediction = int(np.argmax(probs))

            pred_name = ['HOLD', 'BUY', 'SELL'][prediction]
            quality   = self._get_quality_level(len(df))

            if prediction != 0:
                logging.info(
                    f"🎯 [{quality}] [{session}] {pred_name} "
                    f"Conf:{conf:.3f} | B:{probs[1]:.3f} S:{probs[2]:.3f} "
                    f"| Risk R:R:{risk_rr:.2f}")
            elif self.bar_count % 100 == 0:
                logging.debug(
                    f"📊 [{quality}] [{session}] {pred_name} "
                    f"Conf:{conf:.3f} | Bars:{len(df)}")

            return prediction, conf

        except Exception as e:
            logging.exception(f"❌ Prediction error: {e}")
            return 0, 0.0

    def should_enter_trade(
        self,
        prediction:  int,
        confidence:  float,
        bar:         Dict,
        entry_conf:  float = None,
        adx_thresh:  float = 0.0,
    ) -> Tuple[bool, Optional[str]]:
        """
        Determine if entry conditions are met.

        Args:
            prediction:  0=HOLD, 1=BUY, 2=SELL
            confidence:  Model confidence 0-1
            bar:         Current bar data dict
            entry_conf:  Confidence threshold (overrides __init__ value if set)
            adx_thresh:  Unused — kept for BaseStrategy interface compatibility

        Returns:
            (should_enter, direction)
        """
        threshold = entry_conf if entry_conf is not None else self.confidence_threshold
        quality   = self._get_quality_level(self.bar_count)

        if confidence < threshold:
            return False, None

        if prediction == 1:
            logging.info(
                f"✅ [{quality}] BUY SIGNAL | Conf:{confidence:.3f} "
                f"(threshold:{threshold:.2f})")
            logging.info(
                f"   Walk-forward stats at ≥{threshold:.2f}: "
                f"PF {self._get_stats_str(threshold)}")
            return True, 'LONG'

        elif prediction == 2:
            logging.info(
                f"✅ [{quality}] SELL SIGNAL | Conf:{confidence:.3f} "
                f"(threshold:{threshold:.2f})")
            logging.info(
                f"   Walk-forward stats at ≥{threshold:.2f}: "
                f"PF {self._get_stats_str(threshold)}")
            return True, 'SHORT'

        return False, None

    # ══════════════════════════════════════════════════════════
    #   ORB STATE MANAGEMENT
    # ══════════════════════════════════════════════════════════

    def _empty_orb_state(self) -> Dict:
        return {
            'valid':       False,
            'high':        np.nan,
            'low':         np.nan,
            'range':       np.nan,
            'open':        np.nan,
            'volume':      0.0,
            'bar_count':   0,
            'session_date': None,
            'orb_complete': False,
        }

    def _update_orb_state(self, session: str, bar: pd.Series, in_orb_period: bool):
        """Update the ORB state for the given session with a new bar."""
        orb = self._orb_state[session]
        bar_date = bar.name.date()

        # Reset on new session day
        if orb['session_date'] != bar_date:
            self._orb_state[session] = self._empty_orb_state()
            orb = self._orb_state[session]
            orb['session_date'] = bar_date
            orb['open'] = bar['open']

        if in_orb_period and not orb['orb_complete']:
            # Extend ORB range
            if np.isnan(orb['high']):
                orb['high']  = bar['high']
                orb['low']   = bar['low']
            else:
                orb['high']  = max(orb['high'], bar['high'])
                orb['low']   = min(orb['low'],  bar['low'])
            orb['volume']    += bar['volume']
            orb['bar_count'] += 1
            orb['range']      = orb['high'] - orb['low']
            orb['valid']      = True

        elif not in_orb_period and not orb['orb_complete']:
            # ORB period just ended — lock it
            orb['orb_complete'] = True

    def reset_session(self, session: str):
        """Manually reset a session's ORB state (call at session start)."""
        self._orb_state[session] = self._empty_orb_state()

    # ══════════════════════════════════════════════════════════
    #   FEATURE CALCULATION
    # ══════════════════════════════════════════════════════════

    def _add_ffm_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replicate the FFM feature pipeline for live inference.
        These must match the features computed during training exactly.
        Feature names match get_model_feature_columns() from futures_foundation.
        """
        # ── Returns ──
        df['ret_1']  = df['close'].pct_change(1)
        df['ret_3']  = df['close'].pct_change(3)
        df['ret_5']  = df['close'].pct_change(5)
        df['ret_10'] = df['close'].pct_change(10)
        df['ret_20'] = df['close'].pct_change(20)

        # ── ATR (true range) ──
        hl  = df['high'] - df['low']
        hpc = (df['high'] - df['close'].shift(1)).abs()
        lpc = (df['low']  - df['close'].shift(1)).abs()
        tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
        df['atr_14']  = tr.ewm(span=14,  adjust=False).mean()
        df['atr_50']  = tr.ewm(span=50,  adjust=False).mean()
        df['atr_norm'] = df['atr_14'] / (df['close'] + 1e-8)

        # ── Volatility regime ──
        df['vol_regime'] = df['atr_14'] / (df['atr_50'] + 1e-8)

        # ── Volume features ──
        df['vol_ma20']   = df['volume'].rolling(20).mean()
        df['vol_ratio']  = df['volume'] / (df['vol_ma20'] + 1e-8)
        df['vol_ma5']    = df['volume'].rolling(5).mean()
        df['vol_spike']  = df['volume'] / (df['vol_ma5'] + 1e-8)

        # ── Price structure ──
        df['high_20']    = df['high'].rolling(20).max()
        df['low_20']     = df['low'].rolling(20).min()
        df['range_20']   = df['high_20'] - df['low_20']
        df['price_pos']  = (df['close'] - df['low_20']) / (df['range_20'] + 1e-8)

        # ── EMAs ──
        for span in [9, 20, 50, 100, 200]:
            df[f'ema_{span}'] = df['close'].ewm(span=span, adjust=False).mean()

        df['ema9_20_dist']  = (df['ema_9']  - df['ema_20'])  / (df['close'] + 1e-8)
        df['ema20_50_dist'] = (df['ema_20'] - df['ema_50'])  / (df['close'] + 1e-8)
        df['ema50_200_dist']= (df['ema_50'] - df['ema_200']) / (df['close'] + 1e-8)
        df['price_ema20']   = (df['close']  - df['ema_20'])  / (df['atr_14'] + 1e-8)
        df['price_ema50']   = (df['close']  - df['ema_50'])  / (df['atr_14'] + 1e-8)

        # ── Momentum / RSI ──
        delta      = df['close'].diff()
        gain       = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
        loss       = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
        rs         = gain / (loss + 1e-8)
        df['rsi_14'] = 100 - (100 / (1 + rs))
        df['rsi_norm'] = (df['rsi_14'] - 50) / 50

        # ── Bar body / wick features ──
        df['bar_range']  = (df['high'] - df['low']) / (df['atr_14'] + 1e-8)
        body             = (df['close'] - df['open']).abs()
        df['body_ratio'] = body / (df['high'] - df['low'] + 1e-8)
        df['upper_wick'] = (df['high'] - df[['open','close']].max(axis=1)) / (df['atr_14'] + 1e-8)
        df['lower_wick'] = (df[['open','close']].min(axis=1) - df['low'])   / (df['atr_14'] + 1e-8)
        df['bar_dir']    = np.sign(df['close'] - df['open'])

        # ── Rolling returns std ──
        df['ret_std_10'] = df['ret_1'].rolling(10).std()
        df['ret_std_20'] = df['ret_1'].rolling(20).std()

        return df

    def _add_session_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify each bar into a session and compute temporal features."""
        hour = df.index.hour
        minute = df.index.minute

        sess_id   = np.zeros(len(df), dtype=np.int64)
        tod       = np.zeros(len(df), dtype=np.float32)
        in_orb    = np.zeros(len(df), dtype=bool)

        for i, (h, m) in enumerate(zip(hour, minute)):
            bar_min = h * 60 + m

            # NY: 09:30 – 16:00  ORB: 09:30 – 10:30
            if (9*60+30) <= bar_min < (16*60):
                sess_id[i] = 2
                sess_start = 9*60+30
                sess_end   = 16*60
                orb_end    = 10*60+30
                tod[i]     = (bar_min - sess_start) / (sess_end - sess_start)
                in_orb[i]  = bar_min < orb_end

            # London: 03:00 – 08:30  ORB: 03:00 – 04:00
            elif (3*60) <= bar_min < (8*60+30):
                sess_id[i] = 1
                sess_start = 3*60
                sess_end   = 8*60+30
                orb_end    = 4*60
                tod[i]     = (bar_min - sess_start) / (sess_end - sess_start)
                in_orb[i]  = bar_min < orb_end

            # Asia: 18:00 – 01:00 (overnight)
            elif bar_min >= (18*60) or bar_min < (1*60):
                sess_id[i] = 0
                sess_start = 18*60
                orb_end    = 19*60
                if bar_min >= (18*60):
                    elapsed = bar_min - sess_start
                else:
                    elapsed = (24*60 - sess_start) + bar_min
                sess_len = 7*60
                tod[i]   = min(elapsed / sess_len, 1.0)
                in_orb[i] = (bar_min >= (18*60)) and (bar_min < (19*60))

            else:
                sess_id[i] = -1   # off-session

        df['sess_id']          = sess_id
        df['sess_time_of_day'] = tod
        df['in_orb_period']    = in_orb
        df['tmp_day_of_week']  = df.index.dayofweek.astype(np.int64)

        # Temporal cyclical encodings
        df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        df['dow_sin']  = np.sin(2 * np.pi * df.index.dayofweek / 5)
        df['dow_cos']  = np.cos(2 * np.pi * df.index.dayofweek / 5)

        return df

    def _add_orb_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the 20 ORB point-in-time features for every bar.
        These match ORB_FEATURE_COLS used in training.
        """
        sess_map = {0: 'Asia', 1: 'London', 2: 'NY'}

        # Arrays for each ORB feature
        n = len(df)
        orb_high         = np.full(n, np.nan)
        orb_low          = np.full(n, np.nan)
        orb_range        = np.full(n, np.nan)
        orb_range_atr    = np.full(n, np.nan)
        orb_mid          = np.full(n, np.nan)
        price_vs_orb_hi  = np.full(n, np.nan)
        price_vs_orb_lo  = np.full(n, np.nan)
        price_vs_orb_mid = np.full(n, np.nan)
        breakout_dir     = np.zeros(n, dtype=np.float32)
        breakout_strength= np.zeros(n, dtype=np.float32)
        dist_above_hi    = np.zeros(n, dtype=np.float32)
        dist_below_lo    = np.zeros(n, dtype=np.float32)
        in_range         = np.zeros(n, dtype=np.float32)
        vol_ratio_orb    = np.full(n, np.nan)
        orb_vol_norm     = np.full(n, np.nan)
        bars_since_open  = np.zeros(n, dtype=np.float32)
        session_progress = np.zeros(n, dtype=np.float32)
        orb_valid_flag   = np.zeros(n, dtype=np.float32)
        orb_range_pct    = np.full(n, np.nan)
        orb_open_dist    = np.full(n, np.nan)

        for i, (idx, row) in enumerate(df.iterrows()):
            sess_id = int(row['sess_id'])
            if sess_id < 0 or sess_id not in sess_map:
                continue

            sess_name    = sess_map[sess_id]
            in_orb_bar   = bool(row['in_orb_period'])
            atr          = row.get('atr_14', 1.0) or 1.0

            # Update ORB state with this bar
            self._update_orb_state(sess_name, row, in_orb_bar)
            orb = self._orb_state[sess_name]

            if not orb['valid']:
                continue

            oh = orb['high']
            ol = orb['low']
            om = (oh + ol) / 2
            or_ = oh - ol
            close = row['close']

            # Core levels
            orb_high[i]  = oh
            orb_low[i]   = ol
            orb_range[i] = or_
            orb_mid[i]   = om

            orb_range_atr[i]  = or_ / (atr + 1e-8)
            orb_range_pct[i]  = or_ / (close + 1e-8)
            orb_open_dist[i]  = (close - orb['open']) / (atr + 1e-8) if orb['open'] else 0

            # Price vs levels (normalised by ATR)
            price_vs_orb_hi[i]  = (close - oh) / (atr + 1e-8)
            price_vs_orb_lo[i]  = (close - ol) / (atr + 1e-8)
            price_vs_orb_mid[i] = (close - om) / (or_ + 1e-8)

            # Breakout classification
            if close > oh:
                breakout_dir[i]      = 1.0
                dist_above_hi[i]     = (close - oh) / (atr + 1e-8)
                breakout_strength[i] = dist_above_hi[i]
                in_range[i]          = 0.0
            elif close < ol:
                breakout_dir[i]      = -1.0
                dist_below_lo[i]     = (ol - close) / (atr + 1e-8)
                breakout_strength[i] = dist_below_lo[i]
                in_range[i]          = 0.0
            else:
                breakout_dir[i]      = 0.0
                in_range[i]          = 1.0

            # Volume
            if orb['volume'] > 0:
                vol_ratio_orb[i] = row['volume'] / (orb['volume'] / max(orb['bar_count'], 1) + 1e-8)
            vol_ma = row.get('vol_ma20', np.nan)
            if not np.isnan(vol_ma) and vol_ma > 0:
                orb_vol_norm[i] = (orb['volume'] / max(orb['bar_count'], 1)) / vol_ma

            # Session timing
            bso = int(orb['bar_count'])
            sess_prog = float(row.get('sess_time_of_day', 0))
            bars_since_open[i]  = min(bso / 20.0, 1.0)
            session_progress[i] = sess_prog
            orb_valid_flag[i]   = 1.0

        # Assign to DataFrame
        df['orb_high']          = orb_high
        df['orb_low']           = orb_low
        df['orb_range']         = orb_range
        df['orb_range_atr']     = orb_range_atr
        df['orb_mid']           = orb_mid
        df['price_vs_orb_hi']   = price_vs_orb_hi
        df['price_vs_orb_lo']   = price_vs_orb_lo
        df['price_vs_orb_mid']  = price_vs_orb_mid
        df['breakout_dir']      = breakout_dir
        df['breakout_strength'] = breakout_strength
        df['dist_above_hi']     = dist_above_hi
        df['dist_below_lo']     = dist_below_lo
        df['in_orb_range']      = in_range
        df['vol_ratio_orb']     = vol_ratio_orb
        df['orb_vol_norm']      = orb_vol_norm
        df['bars_since_open']   = bars_since_open
        df['session_progress']  = session_progress
        df['orb_valid']         = orb_valid_flag
        df['orb_range_pct']     = orb_range_pct
        df['orb_open_dist']     = orb_open_dist

        return df

    # ══════════════════════════════════════════════════════════
    #   HELPERS
    # ══════════════════════════════════════════════════════════

    def _get_current_session(self, ts: pd.Timestamp) -> Optional[str]:
        """Return session name for a timestamp, or None if off-session."""
        h, m = ts.hour, ts.minute
        bar_min = h * 60 + m
        if (9*60+30) <= bar_min < (16*60):
            return 'NY'
        if (3*60) <= bar_min < (8*60+30):
            return 'London'
        if bar_min >= (18*60) or bar_min < (1*60):
            return 'Asia'
        return None

    def _get_quality_level(self, bars: int) -> str:
        # 64  = minimum to fill the 64-bar sequence window (hard floor)
        # 150 = ATR-14 and vol-MA20 have stabilized (~12 hours)
        # 300 = EMA-50 reliable, all features stable (~1.5 trading days)
        if bars < 64:
            return 'INSUFFICIENT'
        elif bars < 150:
            return 'WARMING_UP'
        elif bars < 300:
            return 'GOOD'
        else:
            return 'EXCELLENT'

    def _get_stats_str(self, threshold: float) -> str:
        """Return walk-forward stats string for a given confidence threshold."""
        stats = {
            0.90: '2.19, WR 31.6%, +31R',
            0.95: '3.25, WR 40.7%, +36R',
            0.80: '1.51, WR 24.1%, +21R',
            0.70: '1.27, WR 21.0%, +13R',
        }
        # Find closest threshold
        closest = min(stats.keys(), key=lambda x: abs(x - threshold))
        return stats[closest]

    def _default_orb_feature_columns(self) -> List[str]:
        return [
            'orb_high', 'orb_low', 'orb_range', 'orb_range_atr', 'orb_mid',
            'price_vs_orb_hi', 'price_vs_orb_lo', 'price_vs_orb_mid',
            'breakout_dir', 'breakout_strength',
            'dist_above_hi', 'dist_below_lo', 'in_orb_range',
            'vol_ratio_orb', 'orb_vol_norm',
            'bars_since_open', 'session_progress', 'orb_valid',
            'orb_range_pct', 'orb_open_dist',
        ]

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()