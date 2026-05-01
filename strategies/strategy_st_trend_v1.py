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
from collections import deque
from typing import Optional, Tuple

from strategies.ffm_strategy_base import FFMStrategyBase

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ── SuperTrend parameters (must match training in st_trend_v1.py) ────────────
ST_PERIOD       = 10
ST_MULT         = 2.0
HTF_PERIOD      = 10
HTF_MULT        = 3.0
SL_ATR_MULT     = 1.5
ATR_RANK_WINDOW = 200


class STTrendStrategyV1(FFMStrategyBase):
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
        super().__init__(model_path, contract_symbol, min_risk_rr,
                         strategy_tag="ST Trend v1")

        # ── 5min SuperTrend state ────────────────────────────────────────────
        self._st_atr: float        = 0.0
        self._st_upper: float      = float('inf')
        self._st_lower: float      = 0.0
        self._st_direction: int    = 1
        self._st_prev_close: float = 0.0
        self._st_initialized: bool = False

        # ── Prior trend tracking ─────────────────────────────────────────────
        self._trend_start_bar: int  = 0
        self._trend_high: float     = 0.0
        self._trend_low: float      = float('inf')
        self._prior_duration: int   = 0
        self._prior_extent: float   = 0.0

        # ── Rolling ATR history for percentile rank ──────────────────────────
        self._atr_history: deque = deque(maxlen=ATR_RANK_WINDOW)

        # ── 1h SuperTrend state ──────────────────────────────────────────────
        self._htf_atr: float         = 0.0
        self._htf_upper: float       = float('inf')
        self._htf_lower: float       = 0.0
        self._htf_direction: int     = 1
        self._htf_prev_close: float  = 0.0
        self._htf_initialized: bool  = False
        self._htf_flip_bar: int      = 0

        self._htf_bar_hour: Optional[int]    = None
        self._htf_bar_date: Optional[object] = None
        self._htf_bar_open: float  = 0.0
        self._htf_bar_high: float  = 0.0
        self._htf_bar_low: float   = float('inf')
        self._htf_bar_close: float = 0.0

        # ── Signal state ─────────────────────────────────────────────────────
        self._is_signal_bar: bool             = False
        self._current_direction: int          = 0
        self._current_st_features: np.ndarray = np.zeros(8, dtype=np.float32)
        self._signal_atr: float               = 0.0

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

    def get_warmup_length(self) -> int:
        return 300

    def on_trade_exit(self, reason: str):
        self._is_signal_bar    = False
        self._current_direction = 0

    # ── FFMStrategyBase abstract hooks ────────────────────────────────────────

    def _is_ready_to_predict(self) -> bool:
        return self._is_signal_bar and self._current_direction != 0

    def _get_strategy_features(self) -> np.ndarray:
        return self._current_st_features

    def _get_signal_direction(self) -> int:
        return self._current_direction

    def _get_candle_types(self, df: pd.DataFrame, seq_len: int) -> np.ndarray:
        return np.zeros((1, seq_len), dtype=np.int64)

    def _get_session_ids(self, df: pd.DataFrame, seq_len: int) -> np.ndarray:
        return np.zeros((1, seq_len), dtype=np.int64)

    def _build_signal_meta(self, confidence: float) -> dict:
        return {
            'confidence':    round(confidence, 4),
            'risk_rr':       round(self._latest_risk_rr, 4),
            'st_direction':  int(self._current_direction),
            'htf_direction': int(self._htf_direction),
            'signal_atr':    round(self._signal_atr, 4),
        }

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
            self._htf_atr        = max(tr, 1e-6)
            hl2                  = (h + l) / 2.0
            self._htf_upper      = hl2 + HTF_MULT * self._htf_atr
            self._htf_lower      = hl2 - HTF_MULT * self._htf_atr
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
            self._st_atr         = max(tr, 1e-6)
            hl2                  = (high + low) / 2.0
            self._st_upper       = hl2 + ST_MULT * self._st_atr
            self._st_lower       = hl2 - ST_MULT * self._st_atr
            self._st_direction   = 1
            self._st_prev_close  = close
            self._trend_start_bar = bar_idx
            self._trend_high     = high
            self._trend_low      = low
            self._st_initialized = True
            self._atr_history.append(self._st_atr)
            return

        tr  = max(high - low, abs(high - self._st_prev_close), abs(low - self._st_prev_close))
        atr = ((ST_PERIOD - 1) * self._st_atr + tr) / ST_PERIOD
        atr = max(atr, 1e-6)

        hl2       = (high + low) / 2.0
        raw_upper = hl2 + ST_MULT * atr
        raw_lower = hl2 - ST_MULT * atr

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
                self._is_signal_bar     = True
                self._current_direction = 1 if direction == 1 else 2
                self._signal_atr        = atr

                st_line   = lower if direction == 1 else upper
                st_dist   = float(np.clip((close - st_line) / atr, -10, 10))
                bar_range = high - low
                flip_body = abs(close - open_) / max(bar_range, 1e-6)
                htf_age   = min((bar_idx - self._htf_flip_bar) / 100.0, 5.0)

                self._current_st_features = np.array([
                    float(direction),
                    st_dist,
                    float(min(self._prior_duration / 100.0, 5.0)),
                    float(min(self._prior_extent / atr, 10.0)),
                    float(atr_rank),
                    float(self._htf_direction),
                    float(htf_age),
                    float(flip_body),
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
            self._trend_high      = high
            self._trend_low       = low
        else:
            # Bull trend tracks only the high; bear trend tracks only the low (matches labeler)
            if direction == 1:
                self._trend_high = max(self._trend_high, high)
            else:
                self._trend_low = min(self._trend_low, low)

        self._st_atr        = atr
        self._st_upper      = upper
        self._st_lower      = lower
        self._st_direction  = direction
        self._st_prev_close = close

    # ── Stop/target ───────────────────────────────────────────────────────────

    def get_stop_target_pts(
        self,
        df: pd.DataFrame,
        direction: str,
        entry_price: float,
    ) -> Tuple[Optional[float], Optional[float]]:
        """Stop: 1.5 × ATR at signal bar. Target: int(predicted_rr) × R (≥2R)."""
        if self._signal_atr <= 0:
            return None, None

        stop_pts = self._signal_atr * SL_ATR_MULT
        raw_rr   = self._latest_risk_rr
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
