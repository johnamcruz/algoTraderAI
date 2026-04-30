#!/usr/bin/env python3
"""
CISD+OTE Strategy v7.0 — FFM Hybrid Transformer

Key differences from v5.1:
  - seq_len: 64 → 96
  - FFM features: 42 custom → 67 via futures_foundation.derive_features()
    Market context (HTF trend, structure, volume, session, EMA, etc.) that lived
    in v5.1's 28-feature CISD vector is now encoded by the backbone sequence.
  - CISD features: 28 → 10 (zone geometry + trade mechanics only)
  - New ONNX input: candle_types (int64 [B, seq_len], 0–5)
  - Input renamed: cisd_features → strategy_features
  - Output: signal_logits (raw logits); confidence is max(softmax) — already computed
  - No session filter: model self-regulates via in_optimal_session CISD feature

Walk-forward thresholds (confidence = max(softmax)):
  0.90 = conservative  |  0.80 = moderate  |  0.70 = aggressive
  See cisd_ote_hybrid_metadata.json for per-fold precision at each threshold.

Requirements:
  pip install onnxruntime
  pip install futures-foundation  (or add futures_foundation/ to PYTHONPATH)

Timestamps must be in Eastern Time for session features to match training.
"""

import logging
import numpy as np
import pandas as pd
import onnxruntime
from collections import deque
from typing import Dict, List, Tuple, Optional

from strategies.strategy_base import BaseStrategy
from futures_foundation import derive_features, get_model_feature_columns, INSTRUMENT_MAP
from utils.bot_utils import parse_future_symbol, MICRO_TO_MINI_MAP

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ── CISD detection constants (must match training in cisd_ote.py) ───────────
SWING_PERIOD        = 6
TOLERANCE           = 0.70
EXPIRY_BARS         = 50
LIQUIDITY_LOOKBACK  = 10
ZONE_MAX_BARS       = 40
FIB_1               = 0.618
FIB_2               = 0.786
HTF_RANGE_BARS      = 96
DISP_BODY_RATIO_MIN = 0.50
DISP_CLOSE_STR_MIN  = 0.60

# ── Risk normalization (must match training) ────────────────────────────────
MAX_RISK_DOLLARS = 300.0
POINT_VALUES = {
    'ES': 50.0, 'NQ': 20.0, 'RTY': 10.0, 'YM': 5.0, 'GC': 100.0,
    'MES': 5.0, 'MNQ': 2.0, 'MRTY': 5.0, 'MYM': 0.50, 'MGC': 10.0,
}

# ── Session constants ────────────────────────────────────────────────────────
OPTIMAL_START_HOUR = 9    # in_optimal_session feature: 09:00–11:00 ET
OPTIMAL_END_HOUR   = 11


class CISDOTEStrategyV7(BaseStrategy):
    """
    CISD+OTE Hybrid Strategy using FFM Transformer backbone (v7.0).

    Architecture:
      FFMBackbone(256-dim, 67 features × 96 bars) + CISDProjection(10 features)
      → signal_logits [B, 2]       apply softmax for signal prob
        risk_predictions [B, 1]
        confidence [B]             max(softmax) — use this for entry_conf

    CISD features (10):
      zone geometry (0–3), direction (4), setup quality (5–6),
      entry timing (7), risk sizing (8), session flag (9)
    """

    def __init__(
        self,
        model_path: str,
        contract_symbol: str,
        min_risk_rr: float = 2.0,
    ):
        super().__init__(model_path, contract_symbol)

        self._instrument = self._resolve_instrument(contract_symbol)
        self._min_risk_rr: float = min_risk_rr

        # CISD zone state
        self._active_zones: deque = deque(maxlen=20)
        self._pivot_highs: deque = deque(maxlen=200)
        self._pivot_lows:  deque = deque(maxlen=200)
        self._last_wicked_high: int = -999
        self._last_wicked_low:  int = -999
        self._bear_pots: deque = deque(maxlen=20)
        self._bull_pots: deque = deque(maxlen=20)

        self._latest_cisd_features: Optional[np.ndarray] = None
        self._latest_zone_bullish: float = 0.0
        self._latest_risk_rr: float = 0.0
        self._latest_signal_meta: dict = {}

        self.skip_stats: dict = {'conf_gate': 0, 'rr_gate': 0, 'hold': 0}

        logging.info("=" * 65)
        logging.info("🎯 CISD+OTE Strategy v7.0 — FFM Hybrid Transformer")
        logging.info("=" * 65)
        logging.info(f"  seq_len=96 | FFM features=67 | CISD features=10")
        logging.info(f"  Optimal session feature: {OPTIMAL_START_HOUR}–{OPTIMAL_END_HOUR} ET")
        logging.info(f"  Recommended threshold: 0.80 (moderate)")
        logging.info(f"  Warmup: ~300 bars")
        if min_risk_rr > 0.0:
            logging.info(f"  RR gate: block when predicted_rr < {min_risk_rr}")
        logging.info("=" * 65)

    # ── BaseStrategy interface ────────────────────────────────────────────────

    @staticmethod
    def _resolve_instrument(contract_symbol: Optional[str]) -> str:
        """Resolve root parent symbol from any contract ID format.

        Handles two formats:
          'CON.F.US.MNQ.M26' (backtest config) → split('.')[-2] → 'MNQ' → MICRO_TO_MINI_MAP → 'NQ'
          'MNQM26' / 'MNQZ5'  (live API name)  → parse_future_symbol → 'NQ'
        """
        if not contract_symbol:
            return ''
        if contract_symbol.count('.') >= 3:
            root = contract_symbol.split('.')[-2].upper()
            return MICRO_TO_MINI_MAP.get(root, root)
        return parse_future_symbol(contract_symbol) or contract_symbol.upper()

    def get_sequence_length(self) -> int:
        return 96

    def get_warmup_length(self) -> int:
        return 200

    def get_feature_columns(self) -> List[str]:
        return self._feature_cols if self._feature_cols is not None else get_model_feature_columns()

    @property
    def active_zone_count(self) -> int:
        return len(self._active_zones)

    def _on_new_bar(self, df: pd.DataFrame, bar_idx: int) -> None:
        self._update_cisd_detector(df, bar_idx)

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute 67 FFM features via derive_features() and update CISD zone state.

        df must have OHLCV columns and a DatetimeIndex in Eastern Time.
        Timestamps drive session_id buckets (0=pre-market, 1=london, 2=ny_am, 3=ny_pm)
        which must match training — derive_features handles this internally.
        """
        self._instrument = self._resolve_instrument(self.contract_symbol) or self._instrument
        df = df.copy()
        if len(df) < 2:
            self._latest_cisd_features = None
            return df

        df = self._compute_ffm_features(df)

        if self._bar_count == 0 and len(df) > 1:
            logging.info(f"⏳ CISD warmup: processing {len(df) - 1} historical bars...")
            self._run_warmup(df)
            logging.info(
                f"✅ CISD warmup done — {len(self._active_zones)} active zone(s), "
                f"{len(self._pivot_highs)} pivot highs"
            )

        self._on_new_bar(df, self._bar_count)
        self._latest_cisd_features = self._build_cisd_feature_vector(df, self._bar_count)
        self._bar_count += 1
        return df

    def load_model(self):
        import os
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        self.model = onnxruntime.InferenceSession(
            self.model_path, providers=['CPUExecutionProvider'])
        input_names = [i.name for i in self.model.get_inputs()]
        expected = {'features', 'strategy_features', 'candle_types',
                    'time_of_day', 'day_of_week', 'instrument_ids', 'session_ids'}
        missing = expected - set(input_names)
        if missing:
            raise ValueError(
                f"Model '{os.path.basename(self.model_path)}' is incompatible with "
                f"CISDOTEStrategyV7. Missing required inputs: {sorted(missing)}. "
                f"Model has: {sorted(input_names)}. "
                f"Use models/cisd_ote_hybrid_v7.onnx with this strategy."
            )
        inputs  = [(i.name, i.shape) for i in self.model.get_inputs()]
        outputs = [(o.name, o.shape) for o in self.model.get_outputs()]
        logging.info(f"  ✅ ONNX loaded: {os.path.basename(self.model_path)}")
        logging.info(f"     Inputs:  {inputs}")
        logging.info(f"     Outputs: {outputs}")
        self._load_feature_cols_from_metadata()

    def predict(self, df: pd.DataFrame) -> Tuple[int, float]:
        """
        Run ONNX inference and return (prediction, confidence).

        prediction:  0=Hold, 1=Buy, 2=Sell
        confidence:  max(softmax(signal_logits)) in [0.5, 1.0]
                     Use this value for entry_conf threshold.

        Direction from zone_is_bullish (CISD feature index 4).
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

            missing = [c for c in feature_cols if c not in df.columns]
            if missing:
                logging.warning(f"⚠️ Missing FFM features: {missing[:5]}...")
                return 0, 0.0

            # ── FFM sequence [1, 96, 67] ────────────────────────────────
            feat_arr = df[feature_cols].values.astype(np.float32)
            feat_arr = np.nan_to_num(feat_arr, nan=0.0, posinf=5.0, neginf=-5.0)
            feat_arr = np.clip(feat_arr, -10.0, 10.0)
            seq = feat_arr[-seq_len:].reshape(1, seq_len, -1)

            # ── CISD strategy features [1, 10] ──────────────────────────
            strategy_features = self._latest_cisd_features.reshape(1, -1).astype(np.float32)

            # ── candle_types [1, 96] — int64 encoding 0–5 ───────────────
            # 0=doji, 1=bull strong, 2=bear strong, 3=bull pin, 4=bear pin, 5=neutral
            if 'candle_type' in df.columns:
                ct = df['candle_type'].fillna(0).values.astype(np.int64)
            else:
                ct = np.zeros(len(df), dtype=np.int64)
            candle_types = ct[-seq_len:].reshape(1, seq_len)

            # ── Temporal inputs — use derive_features columns when available ─
            if 'sess_time_of_day' in df.columns:
                tod = df['sess_time_of_day'].values.astype(np.float32)
            elif hasattr(df.index, 'hour'):
                tod = ((df.index.hour * 60 + df.index.minute) / 1440.0).astype(np.float32)
            else:
                tod = np.zeros(len(df), dtype=np.float32)
            time_of_day = tod[-seq_len:].reshape(1, seq_len)

            # tmp_day_of_week: 0=Mon … 4=Fri (int64)
            if 'tmp_day_of_week' in df.columns:
                dow = df['tmp_day_of_week'].values.astype(np.int64)
            elif hasattr(df.index, 'dayofweek'):
                dow = df.index.dayofweek.values.astype(np.int64)
            else:
                dow = np.zeros(len(df), dtype=np.int64)
            day_of_week = dow[-seq_len:].reshape(1, seq_len)

            # sess_id: 0=pre-market(<3h), 1=london(3–8h), 2=ny_am(8–12h), 3=ny_pm(12–16h) ET
            if 'sess_id' in df.columns:
                sess = df['sess_id'].values.astype(np.int64)
            elif hasattr(df.index, 'hour'):
                h = df.index.hour
                sess = np.where(h < 3, 0,
                       np.where(h < 8, 1,
                       np.where(h < 12, 2, 3))).astype(np.int64)
            else:
                sess = np.full(len(df), 2, dtype=np.int64)
            session_ids = sess[-seq_len:].reshape(1, seq_len)

            # Refresh from live contract_symbol; parse_future_symbol maps micros to parent
            self._instrument = self._resolve_instrument(self.contract_symbol) or self._instrument
            inst_id = INSTRUMENT_MAP.get(self._instrument, 0)
            instrument_ids = np.array([inst_id], dtype=np.int64)

            # ── ONNX inference ───────────────────────────────────────────
            outputs = self.model.run(
                ['signal_logits', 'risk_predictions', 'confidence'],
                {
                    'features':          seq,
                    'strategy_features': strategy_features,
                    'candle_types':      candle_types,
                    'time_of_day':       time_of_day,
                    'day_of_week':       day_of_week,
                    'instrument_ids':    instrument_ids,
                    'session_ids':       session_ids,
                }
            )

            # confidence = max(softmax(signal_logits)), already computed by model
            confidence = float(outputs[2][0])
            self._latest_risk_rr = float(np.array(outputs[1]).flatten()[0])

            # Direction from CISD zone (not from signal class)
            if self._latest_zone_bullish > 0:
                prediction = 1   # BUY
            elif self._latest_zone_bullish < 0:
                prediction = 2   # SELL
            else:
                prediction = 0

            cisd = self._latest_cisd_features
            self._latest_signal_meta = {
                'confidence':          round(confidence, 4),
                'risk_rr':             round(self._latest_risk_rr, 4),
                'zone_is_bullish':     round(float(cisd[4]), 4) if cisd is not None else 0.0,
                'had_liquidity_sweep': round(float(cisd[6]), 4) if cisd is not None else 0.0,
                'entry_distance_pct':  round(float(cisd[7]), 4) if cisd is not None else 0.0,
                'in_optimal_session':  round(float(cisd[9]), 4) if cisd is not None else 0.0,
            }

            logging.debug(
                f"  CISD+OTE v7 | conf={confidence:.3f} "
                f"dir={'BUY' if prediction==1 else 'SELL' if prediction==2 else 'NONE'}"
            )
            return prediction, confidence

        except Exception as e:
            logging.exception(f"❌ CISD+OTE v7 predict error: {e}")
            return 0, 0.0

    def is_trading_allowed(self, timestamp: pd.Timestamp) -> bool:
        """v7.0: no hard session filter — model self-regulates via in_optimal_session."""
        return True

    def should_enter_trade(
        self,
        prediction: int,
        confidence: float,
        bar: Dict,
        entry_conf: float,
    ) -> Tuple[bool, Optional[str]]:
        """
        Gate entry on confidence threshold (max(softmax) in [0.5, 1.0]).

        Recommended thresholds from walk-forward results:
          0.90 = conservative  |  0.80 = moderate  |  0.70 = aggressive
        """
        if confidence < entry_conf:
            self.skip_stats['conf_gate'] += 1
            return False, None

        if self._min_risk_rr > 0.0 and self._latest_risk_rr < self._min_risk_rr:
            logging.info(
                f"🚫 RR gate: predicted_rr={self._latest_risk_rr:.2f} "
                f"< {self._min_risk_rr} — skipping"
            )
            self.skip_stats['rr_gate'] += 1
            return False, None

        if prediction == 1:
            logging.info(f"✅ CISD+OTE v7 BUY  | conf={confidence:.3f} rr={self._latest_risk_rr:.2f}")
            return True, 'LONG'
        elif prediction == 2:
            logging.info(f"✅ CISD+OTE v7 SELL | conf={confidence:.3f} rr={self._latest_risk_rr:.2f}")
            return True, 'SHORT'
        self.skip_stats['hold'] += 1
        return False, None

    def on_trade_exit(self, reason: str):
        if reason == 'STOP_LOSS':
            logging.info(f"🔴 Stop loss — clearing {len(self._active_zones)} zone(s)")
            self._active_zones.clear()
            self._latest_cisd_features = None
            self._latest_zone_bullish = 0.0

    def get_stop_target_pts(self, df, direction, entry_price):
        if not self._active_zones:
            return None, None

        nearest = min(
            self._active_zones,
            key=lambda z: abs(entry_price - (z['fib_top'] + z['fib_bot']) / 2.0)
        )
        ft = nearest['fib_top']
        fb = nearest['fib_bot']

        stop_pts = abs(entry_price - fb) if direction == 'LONG' else abs(ft - entry_price)
        if stop_pts <= 0:
            return None, None

        raw_rr = self._latest_risk_rr
        if raw_rr >= 2.0:
            rr = int(raw_rr)
        else:
            return None, None

        target_pts = stop_pts * rr
        logging.info(
            f"  CISD stop/target | dir={direction} entry={entry_price:.2f} "
            f"zone=[{fb:.2f}–{ft:.2f}] stop={stop_pts:.2f}pts "
            f"target={target_pts:.2f}pts (predicted_rr={raw_rr:.2f} → tier={rr}R)"
        )
        return stop_pts, target_pts

    # ── FFM Feature Computation ───────────────────────────────────────────────

    def _compute_ffm_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Delegate to futures_foundation.derive_features() for all 67 FFM features.

        derive_features expects a 'datetime' column. We move the DatetimeIndex
        to a column, call derive_features, then merge feature values back.
        """
        if isinstance(df.index, pd.DatetimeIndex):
            df_input = df.reset_index().rename(
                columns={df.index.name or 'index': 'datetime'}
            )
        else:
            df_input = df.copy()
            if 'datetime' not in df_input.columns:
                raise ValueError("df must have DatetimeIndex or 'datetime' column")

        feature_df = derive_features(df_input, self._instrument)

        # Copy feature column values back (preserves original df index)
        for col in feature_df.columns:
            df[col] = feature_df[col].values

        return df

    # ── Incremental CISD Zone Detector ────────────────────────────────────────

    def _update_cisd_detector(self, df: pd.DataFrame, abs_bar: int):
        """
        Incremental CISD zone detection using absolute bar indices for stored state.
        Same logic as v5.1 with updated displacement constants.
        """
        n   = len(df)
        bar = n - 1

        if bar < 1:
            return

        o_arr = df['open'].values
        h_arr = df['high'].values
        l_arr = df['low'].values
        c_arr = df['close'].values

        def to_rel(abs_idx):
            return max(0, n - 1 - abs_bar + abs_idx)

        # Track sweep (wick-throughs)
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

        # Pivot confirmation
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

        # Track potential CISD candles
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

        # P/D midpoint
        rng_h = h_arr[max(0, bar - HTF_RANGE_BARS):bar].max() if bar > 0 else h_arr[bar]
        rng_l = l_arr[max(0, bar - HTF_RANGE_BARS):bar].min() if bar > 0 else l_arr[bar]
        pd_mid = (rng_h + rng_l) / 2.0

        # Bearish CISD check
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
                        if br >= DISP_BODY_RATIO_MIN and cs >= DISP_CLOSE_STR_MIN:
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
                                        'is_bullish': False, 'fib_top': fib_top, 'fib_bot': fib_bot,
                                        'created_bar': abs_bar, 'had_sweep': had_sweep,
                                        'disp_strength': float(ratio), 'signal_fired': False,
                                        'entered_zone': False,
                                    }
                                    cisd_dir = 'BEAR'
                            break
            else:
                new_bear.append((pot_price, pot_abs))
        self._bear_pots = new_bear

        # Bullish CISD check
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
                        if br >= DISP_BODY_RATIO_MIN and cs >= DISP_CLOSE_STR_MIN:
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
                                        'is_bullish': True, 'fib_top': fib_top, 'fib_bot': fib_bot,
                                        'created_bar': abs_bar, 'had_sweep': had_sweep,
                                        'disp_strength': float(ratio), 'signal_fired': False,
                                        'entered_zone': False,
                                    }
                                    cisd_dir = 'BULL'
                            break
            else:
                new_bull.append((pot_price, pot_abs))
        self._bull_pots = new_bull

        if cisd_zone is not None:
            self._active_zones.appendleft(cisd_zone)
            logging.info(
                f"🟢 New CISD zone: {'BULL' if cisd_zone['is_bullish'] else 'BEAR'} | "
                f"top={cisd_zone['fib_top']:.2f} bot={cisd_zone['fib_bot']:.2f} | "
                f"total active={len(self._active_zones)}"
            )

        # Expire and update zones
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

        # OTE bounce confirmation
        for z in self._active_zones:
            if z.get('signal_fired') or not z['entered_zone'] or abs_bar <= z['created_bar']:
                continue
            if not (current_low <= z['fib_top'] and current_high >= z['fib_bot']):
                continue
            confirmed = (current_close > current_open if z['is_bullish']
                         else current_close < current_open)
            if confirmed:
                z['signal_fired'] = True
                z['entry_bar']    = abs_bar
                z['entry_price']  = current_close
                break

    # ── CISD Feature Vector (10 features) ────────────────────────────────────

    def _build_cisd_feature_vector(self, df: pd.DataFrame, abs_bar: int) -> Optional[np.ndarray]:
        """
        Build the 10-element CISD feature vector for the current bar.

        Feature order matches CISD_FEATURE_COLS from training (cisd_ote.py):
          0: zone_height_vs_atr     — zone geometry
          1: price_vs_zone_top      — geometry
          2: price_vs_zone_bot      — geometry
          3: zone_age_bars          — how old the setup is
          4: zone_is_bullish        — direction: +1=BUY, -1=SELL
          5: cisd_displacement_str  — quality of displacement candle
          6: had_liquidity_sweep    — sweep before CISD
          7: entry_distance_pct     — depth into OTE zone
          8: risk_dollars_norm      — stop-loss size normalised
          9: in_optimal_session     — 09:00–11:00 ET flag

        All market context features (HTF trend, structure, volume, session
        progress, EMA, etc.) are in the 67-feature backbone sequence.
        """
        if not self._active_zones:
            self._latest_zone_bullish = 0.0
            return None

        c = float(df['close'].iloc[-1])

        # Raw ATR in price units (vty_atr_raw set by derive_features)
        if 'vty_atr_raw' in df.columns:
            atr_raw = float(df['vty_atr_raw'].iloc[-1])
        else:
            h = float(df['high'].iloc[-1]); l_val = float(df['low'].iloc[-1])
            atr_raw = (h - l_val) * 14
        atr_safe = max(atr_raw, 1e-6)

        # Nearest active zone that hasn't been consumed
        nearest = None; nearest_dist = float('inf')
        for z in self._active_zones:
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

        # entry_distance_pct: signed depth into zone (negative = inside zone)
        if z['is_bullish']:
            entry_dist = (c - ft) / zh_safe
        else:
            entry_dist = (fb - c) / zh_safe

        # risk_dollars_norm
        sl = fb if z['is_bullish'] else ft
        risk_pts = abs(c - sl)
        pv = POINT_VALUES.get(self._instrument, 20.0)
        risk_norm = float(np.clip((risk_pts * pv) / MAX_RISK_DOLLARS, 0, 5))

        # in_optimal_session
        in_optimal = 0.0
        if hasattr(df.index, 'hour'):
            h_val = int(df.index[-1].hour)
            in_optimal = 1.0 if (OPTIMAL_START_HOUR <= h_val < OPTIMAL_END_HOUR) else 0.0

        features = np.array([
            np.clip(zh / atr_safe, 0, 10),                  # 0: zone_height_vs_atr
            np.clip((c - ft) / zh_safe, -2, 5),              # 1: price_vs_zone_top
            np.clip((c - fb) / zh_safe, -2, 5),              # 2: price_vs_zone_bot
            np.clip(age / ZONE_MAX_BARS, 0, 5),               # 3: zone_age_bars
            self._latest_zone_bullish,                        # 4: zone_is_bullish
            np.clip(z['disp_strength'], 0, 5),                # 5: cisd_displacement_str
            1.0 if z['had_sweep'] else 0.0,                   # 6: had_liquidity_sweep
            float(np.clip(entry_dist, -2, 5)),                # 7: entry_distance_pct
            risk_norm,                                        # 8: risk_dollars_norm
            in_optimal,                                       # 9: in_optimal_session
        ], dtype=np.float32)

        features = np.nan_to_num(features, nan=0.0, posinf=5.0, neginf=-5.0)
        return np.clip(features, -10.0, 10.0)

    def preprocess_features(self, df: pd.DataFrame) -> np.ndarray:
        """No scaler — FFM features are already normalised by derive_features."""
        arr = df[self.get_feature_columns()].values.astype(np.float32)
        return np.nan_to_num(arr, nan=0.0, posinf=5.0, neginf=-5.0)
