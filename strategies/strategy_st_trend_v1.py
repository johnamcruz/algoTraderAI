#!/usr/bin/env python3
"""
SuperTrend Trend Follow Strategy v1.0 — FFM Hybrid Transformer

Signal: 5-minute SuperTrend(10, 2.0) flip + 1h SuperTrend(10, 3.0) HTF alignment
Stop:   1.5 × ATR (Wilder's ATR, period=10) — dynamic sizing via bot's risk_amount
TP:     Risk head floor — <2R skip, ≥2R → int(predicted_rr) × R

Architecture identical to CISD+OTE v7:
  FFMBackbone(256-dim, 67 features × 96 bars) + STProjection(8 features)
  → signal_logits [B, 2], risk_predictions [B, 1], confidence [B]

Model: st_trend_v11.onnx (Phase 2: signal head + calibrated risk head)

Walk-forward thresholds (confidence = max(softmax)):
  0.90 = conservative (recommended start)
  0.85 = moderate (after live validation)
  0.80 = aggressive

Inference only fires on signal bars (ST flip + HTF aligned). All other bars
return (0, 0.0) immediately without running the model.
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


# ── SuperTrend parameters (must match training in st_trend_v1.py) ────────────
ST_PERIOD       = 10
ST_MULT         = 2.0
HTF_PERIOD      = 10
HTF_MULT        = 3.0
SL_ATR_MULT     = 1.5
ATR_RANK_WINDOW = 200

# ── Point values (micro contracts) ───────────────────────────────────────────
POINT_VALUES = {
    'ES': 50.0, 'NQ': 20.0, 'RTY': 10.0, 'YM': 5.0, 'GC': 100.0,
    'MES': 5.0, 'MNQ': 2.0, 'MRTY': 5.0, 'MYM': 0.50, 'MGC': 10.0,
}


class STTrendStrategyV1(BaseStrategy):
    """
    SuperTrend Trend Follow using FFM Transformer backbone (v1.0).

    ST features (8) — index order must match training:
      0: st_direction              +1=bull, -1=bear
      1: st_line_distance          (close - ST_line) / ATR, clipped [-10, 10]
      2: prior_trend_duration_norm bars in prior trend / 100, capped 5
      3: prior_trend_extent_norm   prior range / ATR, capped 10
      4: atr_rank_pct              rolling 200-bar ATR percentile [0, 1]
      5: htf_st_direction          1h ST direction +1/-1
      6: htf_st_age_norm           5m bars since last 1h flip / 100, capped 5
      7: flip_bar_body_pct         |close-open| / (high-low) at flip bar [0, 1]
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

        # ── 5min SuperTrend state ────────────────────────────────────────────
        self._st_atr: float       = 0.0
        self._st_upper: float     = float('inf')
        self._st_lower: float     = 0.0
        self._st_direction: int   = 1          # init bull per spec
        self._st_prev_close: float = 0.0
        self._st_initialized: bool = False

        # ── Prior trend tracking ─────────────────────────────────────────────
        self._trend_start_bar: int   = 0
        self._trend_high: float      = 0.0
        self._trend_low: float       = float('inf')
        self._prior_duration: int    = 0
        self._prior_extent: float    = 0.0

        # ── Rolling ATR history for percentile rank ──────────────────────────
        self._atr_history: deque = deque(maxlen=ATR_RANK_WINDOW)

        # ── 1h SuperTrend state ──────────────────────────────────────────────
        self._htf_atr: float        = 0.0
        self._htf_upper: float      = float('inf')
        self._htf_lower: float      = 0.0
        self._htf_direction: int    = 1
        self._htf_prev_close: float = 0.0
        self._htf_initialized: bool = False
        self._htf_flip_bar: int     = 0        # 5m bar index of last 1h flip

        # Current in-progress 1h bar (aggregated from 5m bars)
        self._htf_bar_hour: Optional[int]    = None
        self._htf_bar_date: Optional[object] = None
        self._htf_bar_open: float  = 0.0
        self._htf_bar_high: float  = 0.0
        self._htf_bar_low: float   = float('inf')
        self._htf_bar_close: float = 0.0

        # ── Signal state ─────────────────────────────────────────────────────
        self._is_signal_bar: bool          = False
        self._current_direction: int       = 0   # 0=none, 1=buy, 2=sell
        self._current_st_features: np.ndarray = np.zeros(8, dtype=np.float32)
        self._signal_atr: float            = 0.0  # ATR at signal bar for stop sizing
        self._latest_risk_rr: float        = 0.0
        self._latest_signal_meta: dict     = {}

        self.skip_stats: dict = {'conf_gate': 0, 'rr_gate': 0, 'hold': 0}

        logging.info("=" * 65)
        logging.info("📈 SuperTrend Strategy v1.0 — FFM Hybrid Transformer")
        logging.info("=" * 65)
        logging.info(f"  seq_len=96 | FFM features=67 | ST features=8")
        logging.info(f"  5m ST({ST_PERIOD}, {ST_MULT}) + 1h ST({HTF_PERIOD}, {HTF_MULT})")
        logging.info(f"  SL: {SL_ATR_MULT}×ATR | Recommended threshold: 0.90")
        if min_risk_rr > 0.0:
            logging.info(f"  RR gate: block when predicted_rr < {min_risk_rr}")
        logging.info("=" * 65)

    # ── BaseStrategy interface ────────────────────────────────────────────────

    @staticmethod
    def _resolve_instrument(contract_symbol: Optional[str]) -> str:
        if not contract_symbol:
            return ''
        if contract_symbol.count('.') >= 3:
            root = contract_symbol.split('.')[-2].upper()
            return MICRO_TO_MINI_MAP.get(root, root)
        return parse_future_symbol(contract_symbol) or contract_symbol.upper()

    def get_sequence_length(self) -> int:
        return 96

    def get_warmup_length(self) -> int:
        return 300

    def get_feature_columns(self) -> List[str]:
        return get_model_feature_columns()

    def is_trading_allowed(self, timestamp: pd.Timestamp) -> bool:
        return True

    def on_trade_exit(self, reason: str):
        self._is_signal_bar = False
        self._current_direction = 0

    # ── Incremental bar processing ────────────────────────────────────────────

    def _on_new_bar(self, df: pd.DataFrame, bar_idx: int) -> None:
        if len(df) < 1:
            return
        row   = df.iloc[-1]
        high  = float(row['high'])
        low   = float(row['low'])
        close = float(row['close'])
        open_ = float(row['open'])
        ts    = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else None

        self._update_htf(high, low, close, ts, bar_idx)
        self._update_st_5m(high, low, close, open_, bar_idx)

    def _update_htf(
        self,
        high: float, low: float, close: float,
        ts: Optional[pd.Timestamp],
        bar_idx: int,
    ) -> None:
        """Aggregate 5m bars into 1h bars and update 1h Wilder's ATR SuperTrend."""
        if ts is None:
            return

        bar_hour = ts.hour
        bar_date = ts.date()

        if self._htf_bar_hour is None:
            self._htf_bar_hour  = bar_hour
            self._htf_bar_date  = bar_date
            self._htf_bar_open  = close
            self._htf_bar_high  = high
            self._htf_bar_low   = low
            self._htf_bar_close = close
            return

        hour_changed = (bar_hour != self._htf_bar_hour or bar_date != self._htf_bar_date)
        if hour_changed:
            self._finalize_htf_bar(bar_idx)
            self._htf_bar_hour  = bar_hour
            self._htf_bar_date  = bar_date
            self._htf_bar_open  = close
            self._htf_bar_high  = high
            self._htf_bar_low   = low
            self._htf_bar_close = close
        else:
            self._htf_bar_high  = max(self._htf_bar_high, high)
            self._htf_bar_low   = min(self._htf_bar_low, low)
            self._htf_bar_close = close

    def _finalize_htf_bar(self, bar_idx: int) -> None:
        """Finalize completed 1h bar and run one step of Wilder's ATR SuperTrend."""
        h = self._htf_bar_high
        l = self._htf_bar_low
        c = self._htf_bar_close

        if not self._htf_initialized:
            tr = h - l
            self._htf_atr   = max(tr, 1e-6)
            hl2             = (h + l) / 2.0
            self._htf_upper = hl2 + HTF_MULT * self._htf_atr
            self._htf_lower = hl2 - HTF_MULT * self._htf_atr
            self._htf_direction  = 1
            self._htf_prev_close = c
            self._htf_initialized = True
            return

        tr  = max(h - l, abs(h - self._htf_prev_close), abs(l - self._htf_prev_close))
        atr = ((HTF_PERIOD - 1) * self._htf_atr + tr) / HTF_PERIOD
        atr = max(atr, 1e-6)

        hl2       = (h + l) / 2.0
        raw_upper = hl2 + HTF_MULT * atr
        raw_lower = hl2 - HTF_MULT * atr

        # Save previous bar's clamped bands — direction check must use these (matches training)
        prev_upper = self._htf_upper
        prev_lower = self._htf_lower

        upper = raw_upper if (raw_upper < prev_upper or self._htf_prev_close > prev_upper) else prev_upper
        lower = raw_lower if (raw_lower > prev_lower or self._htf_prev_close < prev_lower) else prev_lower

        prev_dir = self._htf_direction
        if prev_dir == -1:
            direction = 1 if c > prev_upper else -1
        else:
            direction = -1 if c < prev_lower else 1

        if direction != prev_dir:
            self._htf_flip_bar = bar_idx
            logging.debug(f"1h ST flip: {'BULL' if direction == 1 else 'BEAR'} @ bar {bar_idx}")

        self._htf_atr        = atr
        self._htf_upper      = upper
        self._htf_lower      = lower
        self._htf_direction  = direction
        self._htf_prev_close = c

    def _update_st_5m(
        self,
        high: float, low: float, close: float, open_: float,
        bar_idx: int,
    ) -> None:
        """Update 5m Wilder's ATR SuperTrend and detect flip signals."""
        self._is_signal_bar = False

        if not self._st_initialized:
            tr = high - low
            self._st_atr        = max(tr, 1e-6)
            hl2                 = (high + low) / 2.0
            self._st_upper      = hl2 + ST_MULT * self._st_atr
            self._st_lower      = hl2 - ST_MULT * self._st_atr
            self._st_direction  = 1
            self._st_prev_close = close
            self._trend_start_bar = bar_idx
            self._trend_high    = high
            self._trend_low     = low
            self._st_initialized = True
            self._atr_history.append(self._st_atr)
            return

        tr  = max(high - low, abs(high - self._st_prev_close), abs(low - self._st_prev_close))
        atr = ((ST_PERIOD - 1) * self._st_atr + tr) / ST_PERIOD
        atr = max(atr, 1e-6)

        hl2       = (high + low) / 2.0
        raw_upper = hl2 + ST_MULT * atr
        raw_lower = hl2 - ST_MULT * atr

        # Save previous bar's clamped bands — direction check must use these (matches training)
        prev_upper = self._st_upper
        prev_lower = self._st_lower

        upper = raw_upper if (raw_upper < prev_upper or self._st_prev_close > prev_upper) else prev_upper
        lower = raw_lower if (raw_lower > prev_lower or self._st_prev_close < prev_lower) else prev_lower

        prev_dir = self._st_direction
        if prev_dir == -1:
            direction = 1 if close > prev_upper else -1
        else:
            direction = -1 if close < prev_lower else 1

        # Compute rank BEFORE appending — training uses window[start:i] which excludes bar i
        atr_rank = sum(1 for a in self._atr_history if a < atr) / max(len(self._atr_history), 1)
        self._atr_history.append(atr)

        is_flip = (direction != prev_dir)
        if is_flip:
            self._prior_duration = bar_idx - self._trend_start_bar
            self._prior_extent   = self._trend_high - self._trend_low

            if self._htf_direction == direction:
                # ST flip + HTF aligned → signal
                self._is_signal_bar    = True
                self._current_direction = 1 if direction == 1 else 2
                self._signal_atr       = atr

                st_line    = lower if direction == 1 else upper
                st_dist    = float(np.clip((close - st_line) / atr, -10, 10))
                bar_range  = high - low
                flip_body  = abs(close - open_) / max(bar_range, 1e-6)
                htf_age    = min((bar_idx - self._htf_flip_bar) / 100.0, 5.0)

                self._current_st_features = np.array([
                    float(direction),                                    # 0: st_direction
                    st_dist,                                             # 1: st_line_distance
                    float(min(self._prior_duration / 100.0, 5.0)),       # 2: prior_trend_duration_norm
                    float(min(self._prior_extent / atr, 10.0)),          # 3: prior_trend_extent_norm
                    float(atr_rank),                                     # 4: atr_rank_pct
                    float(self._htf_direction),                          # 5: htf_st_direction
                    float(htf_age),                                      # 6: htf_st_age_norm
                    float(flip_body),                                    # 7: flip_bar_body_pct
                ], dtype=np.float32)

                logging.info(
                    f"🟢 ST flip: {'BULL' if direction == 1 else 'BEAR'} | "
                    f"HTF={'BULL' if self._htf_direction == 1 else 'BEAR'} ✅ signal"
                )
            else:
                self._current_direction = 0
                logging.debug(
                    f"ST flip: {'BULL' if direction == 1 else 'BEAR'} | "
                    f"HTF={'BULL' if self._htf_direction == 1 else 'BEAR'} — HTF mismatch, skipping"
                )

            # Reset trend tracking for the new trend regardless of HTF alignment
            self._trend_start_bar = bar_idx
            self._trend_high = high
            self._trend_low  = low
        else:
            # Only extend the dimension relevant to the current trend — matches training labeler:
            # bull trend tracks only the high; bear trend tracks only the low.
            if direction == 1:
                self._trend_high = max(self._trend_high, high)
            else:
                self._trend_low = min(self._trend_low, low)

        self._st_atr        = atr
        self._st_upper      = upper
        self._st_lower      = lower
        self._st_direction  = direction
        self._st_prev_close = close

    # ── FFM feature computation ───────────────────────────────────────────────

    def _compute_ffm_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(df.index, pd.DatetimeIndex):
            df_input = df.reset_index().rename(
                columns={df.index.name or 'index': 'datetime'}
            )
        else:
            df_input = df.copy()
            if 'datetime' not in df_input.columns:
                raise ValueError("df must have DatetimeIndex or 'datetime' column")

        feature_df = derive_features(df_input, self._instrument)
        for col in feature_df.columns:
            df[col] = feature_df[col].values
        return df

    # ── add_features ─────────────────────────────────────────────────────────

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        self._instrument = self._resolve_instrument(self.contract_symbol) or self._instrument
        df = df.copy()
        if len(df) < 2:
            return df

        df = self._compute_ffm_features(df)

        if self._bar_count == 0 and len(df) > 1:
            logging.info(f"⏳ ST warmup: processing {len(df) - 1} historical bars...")
            self._run_warmup(df)
            logging.info("✅ ST warmup done")

        self._on_new_bar(df, self._bar_count)
        self._bar_count += 1
        return df

    # ── Model ─────────────────────────────────────────────────────────────────

    def load_model(self):
        import os
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        self.model = onnxruntime.InferenceSession(
            self.model_path, providers=['CPUExecutionProvider'])
        input_names = {i.name for i in self.model.get_inputs()}
        expected    = {'features', 'strategy_features', 'candle_types',
                       'time_of_day', 'day_of_week', 'instrument_ids', 'session_ids'}
        missing = expected - input_names
        if missing:
            raise ValueError(
                f"Model incompatible with STTrendStrategyV1. "
                f"Missing inputs: {sorted(missing)}. Use st_trend_v11.onnx."
            )
        inputs  = [(i.name, i.shape) for i in self.model.get_inputs()]
        outputs = [(o.name, o.shape) for o in self.model.get_outputs()]
        logging.info(f"  ✅ ONNX loaded: {os.path.basename(self.model_path)}")
        logging.info(f"     Inputs:  {inputs}")
        logging.info(f"     Outputs: {outputs}")

    def predict(self, df: pd.DataFrame) -> Tuple[int, float]:
        """
        Run ONNX inference only on ST flip bars (HTF-aligned).
        Non-signal bars return (0, 0.0) immediately.
        """
        try:
            if not self._is_signal_bar or self._current_direction == 0:
                return 0, 0.0

            seq_len      = self.get_sequence_length()
            feature_cols = self.get_feature_columns()

            if df.empty or len(df) < seq_len:
                logging.debug(f"⏳ Warmup: {len(df)}/{seq_len} bars")
                return 0, 0.0

            missing = [c for c in feature_cols if c not in df.columns]
            if missing:
                logging.warning(f"⚠️ Missing FFM features: {missing[:5]}...")
                return 0, 0.0

            # FFM sequence [1, 96, 67]
            feat_arr = df[feature_cols].values.astype(np.float32)
            feat_arr = np.nan_to_num(feat_arr, nan=0.0, posinf=5.0, neginf=-5.0)
            feat_arr = np.clip(feat_arr, -10.0, 10.0)
            seq = feat_arr[-seq_len:].reshape(1, seq_len, -1)

            # ST strategy features [1, 8]
            strategy_features = self._current_st_features.reshape(1, -1).astype(np.float32)

            # candle_types — all zeros per spec
            candle_types = np.zeros((1, seq_len), dtype=np.int64)

            # time_of_day [1, 96]
            if 'sess_time_of_day' in df.columns:
                tod = df['sess_time_of_day'].values.astype(np.float32)
            elif hasattr(df.index, 'hour'):
                tod = ((df.index.hour * 60 + df.index.minute) / 1440.0).astype(np.float32)
            else:
                tod = np.zeros(len(df), dtype=np.float32)
            time_of_day = tod[-seq_len:].reshape(1, seq_len)

            # day_of_week [1, 96]
            if 'tmp_day_of_week' in df.columns:
                dow = df['tmp_day_of_week'].values.astype(np.int64)
            elif hasattr(df.index, 'dayofweek'):
                dow = df.index.dayofweek.values.astype(np.int64)
            else:
                dow = np.zeros(len(df), dtype=np.int64)
            day_of_week = dow[-seq_len:].reshape(1, seq_len)

            # session_ids — all zeros per spec
            session_ids = np.zeros((1, seq_len), dtype=np.int64)

            # instrument_ids
            self._instrument = self._resolve_instrument(self.contract_symbol) or self._instrument
            inst_id          = INSTRUMENT_MAP.get(self._instrument, 0)
            instrument_ids   = np.array([inst_id], dtype=np.int64)

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

            confidence           = float(outputs[2][0])
            self._latest_risk_rr = float(np.array(outputs[1]).flatten()[0])

            self._latest_signal_meta = {
                'confidence':    round(confidence, 4),
                'risk_rr':       round(self._latest_risk_rr, 4),
                'st_direction':  int(self._current_direction),
                'htf_direction': int(self._htf_direction),
                'signal_atr':    round(self._signal_atr, 4),
            }

            logging.debug(
                f"  ST v1 | conf={confidence:.3f} "
                f"dir={'BUY' if self._current_direction == 1 else 'SELL'}"
            )
            return self._current_direction, confidence

        except Exception as e:
            logging.exception(f"❌ ST Trend v1 predict error: {e}")
            return 0, 0.0

    def should_enter_trade(
        self,
        prediction: int,
        confidence: float,
        bar: Dict,
        entry_conf: float,
        adx_thresh: float,
    ) -> Tuple[bool, Optional[str]]:
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
            logging.info(f"✅ ST Trend v1 BUY  | conf={confidence:.3f} rr={self._latest_risk_rr:.2f}")
            return True, 'LONG'
        if prediction == 2:
            logging.info(f"✅ ST Trend v1 SELL | conf={confidence:.3f} rr={self._latest_risk_rr:.2f}")
            return True, 'SHORT'

        self.skip_stats['hold'] += 1
        return False, None

    def get_stop_target_pts(
        self,
        df: pd.DataFrame,
        direction: str,
        entry_price: float,
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Stop: 1.5 × ATR (Wilder's, period=10) captured at signal bar.
        Target: int(predicted_rr) × R — skips if predicted_rr < 2.0.
        Bot sizes contracts as floor(risk_amount / (stop_pts × point_value)).
        """
        if self._signal_atr <= 0:
            return None, None

        stop_pts = self._signal_atr * SL_ATR_MULT

        raw_rr = self._latest_risk_rr
        if raw_rr >= 2.0:
            rr = int(raw_rr)
        else:
            return None, None

        target_pts = stop_pts * rr
        logging.info(
            f"  ST stop/target | dir={direction} entry={entry_price:.2f} "
            f"stop={stop_pts:.2f}pts target={target_pts:.2f}pts "
            f"(predicted_rr={raw_rr:.2f} → tier={rr}R)"
        )
        return stop_pts, target_pts

    def preprocess_features(self, df: pd.DataFrame) -> np.ndarray:
        arr = df[self.get_feature_columns()].values.astype(np.float32)
        return np.nan_to_num(arr, nan=0.0, posinf=5.0, neginf=-5.0)
