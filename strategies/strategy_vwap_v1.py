#!/usr/bin/env python3
"""
VWAP Reversion Strategy v1.1 — FFM Hybrid Transformer

Signal: 5-min bar closes >= 2.0 SDs from daily VWAP + reversal candle
Stop:   bar extreme (low/high) ± ATR × 0.1  (exact match with training labeler)
TP:     fixed 2R (matches the binary training label — "did price reach 2R within 96 bars?")
        predicted_rr is the risk head's max-excursion estimate; use as a gate (≥4.0),
        not as a TP multiplier. At conf≥0.70 + rr≥4.0: ~93% expected win rate.
        Default min_risk_rr=4.0 is baked into the constructor; no CLI flag needed.

Architecture identical to CISD+OTE v7 and SuperTrend v1:
  FFMBackbone(256-dim, 67 features × 96 bars) + VWAPProjection(8 features)
  → signal_logits [B, 2], risk_predictions [B, 1], confidence [B]

Inference only fires on VWAP signal bars. All other bars return (0, 0.0).

Walk-forward thresholds (v1.1, held-out test sets):
  0.70 = recommended (94.1% precision)
  0.80 = conservative (95.7% precision)

Session: 07:00–16:00 Eastern
"""

import logging
import numpy as np
import pandas as pd
from collections import deque
from typing import Optional, Tuple

from strategies.ffm_strategy_base import FFMStrategyBase

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ── VWAP parameters (must match vwap_v1.py training / v1.1 model context) ───
VWAP_DEV_THRESH  = 2.0
SL_ATR_MULT      = 0.1
ATR_PERIOD       = 14
ATR_RANK_WINDOW  = 200
SD_RANK_WINDOW   = 500
SESSION_START    = 7
SESSION_END      = 16

VWAP_FEATURE_COLS = [
    'vwap_dev_sd',
    'vwap_dev_pct',
    'atr_rank_pct',
    'htf_vwap_dev_sd',
    'daily_vwap_trend',
    'bars_at_extreme',
    'dev_momentum',
    'vwap_sd_rank',
]


class VWAPReversionStrategyV1(FFMStrategyBase):
    """
    VWAP Reversion using FFM Transformer backbone (v1.1).

    VWAP strategy features (8) — index order must match vwap_v1.py training:
      0: vwap_dev_sd        signed SD deviation from daily VWAP
      1: vwap_dev_pct       pct distance from VWAP, signed
      2: atr_rank_pct       rolling 200-bar ATR percentile
      3: htf_vwap_dev_sd    1H-bucket VWAP deviation in SDs
      4: daily_vwap_trend   +1 VWAP rising, -1 falling, 0 flat
      5: bars_at_extreme    consecutive bars at extreme, capped at 20
      6: dev_momentum       dev_sd[i] - dev_sd[i-3]
      7: vwap_sd_rank       rolling 500-bar VWAP-SD percentile
    """

    def __init__(
        self,
        model_path: str,
        contract_symbol: str,
        min_risk_rr: float = 4.0,
    ):
        super().__init__(model_path, contract_symbol, min_risk_rr,
                         strategy_tag="VWAP Rev")

        # ── Daily VWAP state (resets at midnight ET each trading day) ────────
        self._current_date: Optional[object] = None
        self._cum_tp_vol: float  = 0.0
        self._cum_vol: float     = 0.0
        self._vwap_devs: list    = []
        self._vwap: float        = 0.0
        self._vwap_sd: float     = 0.0

        # ── 1H bucket VWAP (resets each calendar hour) ───────────────────────
        self._htf_hour_key: Optional[tuple] = None
        self._htf_cum_tv: float = 0.0
        self._htf_cum_v: float  = 0.0
        self._htf_devs: list    = []
        self._htf_vwap_dev_sd: float = 0.0

        # ── Wilder ATR (14-period) ────────────────────────────────────────────
        self._atr: float            = 0.0
        self._prev_close: float     = 0.0
        self._atr_initialized: bool = False

        # ── Rolling percentile windows ────────────────────────────────────────
        self._atr_history: deque = deque(maxlen=ATR_RANK_WINDOW)
        self._sd_history: deque  = deque(maxlen=SD_RANK_WINDOW)

        # ── Lookback deques for momentum / slope features ─────────────────────
        self._recent_dev_sd: deque = deque(maxlen=4)
        self._recent_vwap: deque   = deque(maxlen=13)

        # ── Running consecutive-bars counter ──────────────────────────────────
        self._bars_at_extreme: int = 0

        # ── Signal state ──────────────────────────────────────────────────────
        self._is_signal_bar: bool               = False
        self._current_direction: int            = 0
        self._current_vwap_features: np.ndarray = np.zeros(8, dtype=np.float32)
        self._signal_atr: float                 = 0.0
        self._signal_sl_dist: float             = 0.0
        self._signal_vwap: float                = 0.0

        logging.info("=" * 65)
        logging.info("📊 VWAP Reversion Strategy v1.1 — FFM Hybrid Transformer")
        logging.info("=" * 65)
        logging.info(f"  seq_len=96 | FFM features=67 | VWAP features=8")
        logging.info(f"  Trigger: |dev_sd| >= {VWAP_DEV_THRESH} SDs + reversal candle")
        logging.info(f"  SL: bar_extreme ± {SL_ATR_MULT}×ATR | Session: {SESSION_START}h–{SESSION_END}h ET")
        logging.info(f"  TP: <2R skip | 2–4R fixed 2R | ≥4R → VWAP distance (mean-reversion anchor)")
        logging.info(f"  Recommended threshold: 0.70")
        if min_risk_rr > 0.0:
            logging.info(f"  RR gate: block when predicted_rr < {min_risk_rr}")
        logging.info("=" * 65)

    # ── BaseStrategy interface ────────────────────────────────────────────────

    def get_warmup_length(self) -> int:
        return 200

    def on_trade_exit(self, reason: str):
        self._is_signal_bar     = False
        self._current_direction = 0

    # ── FFMStrategyBase abstract hooks ────────────────────────────────────────

    def _is_ready_to_predict(self) -> bool:
        return self._is_signal_bar and self._current_direction != 0

    def _get_strategy_features(self) -> np.ndarray:
        return self._current_vwap_features

    def _get_signal_direction(self) -> int:
        return self._current_direction

    def _build_signal_meta(self, confidence: float) -> dict:
        return {
            'confidence':   round(confidence, 4),
            'risk_rr':      round(self._latest_risk_rr, 4),
            'direction':    'BUY' if self._current_direction == 1 else 'SELL',
            'vwap_dev_sd':  round(float(self._current_vwap_features[0]), 4),
            'dev_momentum': round(float(self._current_vwap_features[6]), 4),
            'sl_dist':      round(self._signal_sl_dist, 4),
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
        vol   = float(row.get('volume', 0.0) or 0.0)
        ts    = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else None

        self._is_signal_bar = False

        # ── Wilder ATR (14-period) ────────────────────────────────────────────
        if not self._atr_initialized:
            self._atr             = max(high - low, 1e-6)
            self._prev_close      = close
            self._atr_initialized = True
        else:
            tr         = max(high - low,
                             abs(high - self._prev_close),
                             abs(low  - self._prev_close))
            self._atr  = max(((ATR_PERIOD - 1) * self._atr + tr) / ATR_PERIOD, 1e-6)
            self._prev_close = close

        # ATR rank: compute BEFORE appending (matches training labeler)
        atr_rank = (sum(1 for a in self._atr_history if a < self._atr)
                    / max(len(self._atr_history), 1))
        self._atr_history.append(self._atr)

        # ── Daily VWAP (resets on calendar date change) ───────────────────────
        bar_date = ts.date() if ts is not None else None
        if bar_date != self._current_date:
            self._current_date = bar_date
            self._cum_tp_vol   = 0.0
            self._cum_vol      = 0.0
            self._vwap_devs    = []

        tp              = (high + low + close) / 3.0
        vol_pos          = max(vol, 0.0)
        self._cum_tp_vol += tp * vol_pos
        self._cum_vol    += vol_pos
        self._vwap       = self._cum_tp_vol / self._cum_vol if self._cum_vol > 0 else tp
        self._vwap_devs.append(tp - self._vwap)
        self._vwap_sd    = float(np.std(self._vwap_devs)) if len(self._vwap_devs) > 2 else 0.0

        # VWAP-SD rank: compute BEFORE appending
        if self._vwap_sd > 0:
            sd_rank = (sum(1 for s in self._sd_history if s < self._vwap_sd)
                       / max(len(self._sd_history), 1))
            self._sd_history.append(self._vwap_sd)
        else:
            sd_rank = 0.0

        # ── 1H bucket VWAP (resets each calendar hour) ───────────────────────
        hour_key = ((ts.year, ts.month, ts.day, ts.hour) if ts is not None else None)
        if hour_key != self._htf_hour_key:
            self._htf_hour_key = hour_key
            self._htf_cum_tv   = 0.0
            self._htf_cum_v    = 0.0
            self._htf_devs     = []

        self._htf_cum_tv += tp * vol_pos
        self._htf_cum_v  += vol_pos
        vwap_1h           = self._htf_cum_tv / self._htf_cum_v if self._htf_cum_v > 0 else tp
        self._htf_devs.append(tp - vwap_1h)
        sd_1h             = float(np.std(self._htf_devs)) if len(self._htf_devs) > 2 else 0.0
        self._htf_vwap_dev_sd = (
            float(np.clip((tp - vwap_1h) / (sd_1h + 1e-6), -5, 5)) if sd_1h > 0 else 0.0
        )

        # ── Signed deviation from daily VWAP ─────────────────────────────────
        dev_sd = (close - self._vwap) / (self._vwap_sd + 1e-6) if self._vwap_sd > 0 else 0.0

        # ── Momentum: dev_sd[i] - dev_sd[i-3] ────────────────────────────────
        self._recent_dev_sd.append(dev_sd)
        oldest_dev = list(self._recent_dev_sd)[0] if len(self._recent_dev_sd) == 4 else dev_sd
        dev_mom    = float(np.clip(dev_sd - oldest_dev, -5, 5))

        # ── VWAP slope over last 12 bars ──────────────────────────────────────
        self._recent_vwap.append(self._vwap)
        oldest_vwap = list(self._recent_vwap)[0] if len(self._recent_vwap) == 13 else self._vwap
        vwap_trend  = float(np.sign(self._vwap - oldest_vwap))

        # ── Consecutive bars at extreme (capped 20) ───────────────────────────
        if self._vwap_sd > 0 and abs(dev_sd) >= VWAP_DEV_THRESH:
            self._bars_at_extreme = min(self._bars_at_extreme + 1, 20)
        else:
            self._bars_at_extreme = 0

        # ── Signal detection ──────────────────────────────────────────────────
        if self._vwap_sd <= 0 or abs(dev_sd) < VWAP_DEV_THRESH:
            return

        is_bull = (dev_sd < 0 and close > open_)
        is_bear = (dev_sd > 0 and close < open_)
        if not is_bull and not is_bear:
            return

        sl_dist = abs(close - (low if is_bull else high)) + SL_ATR_MULT * self._atr
        if sl_dist <= 0:
            return

        self._is_signal_bar     = True
        self._current_direction = 1 if is_bull else 2
        self._signal_atr        = self._atr
        self._signal_sl_dist    = sl_dist
        self._signal_vwap       = self._vwap

        self._current_vwap_features = np.array([
            float(dev_sd),
            float((close - self._vwap) / (close + 1e-6) * 100),
            float(atr_rank),
            float(self._htf_vwap_dev_sd),
            float(vwap_trend),
            float(self._bars_at_extreme),
            float(dev_mom),
            float(sd_rank),
        ], dtype=np.float32)

        logging.info(
            f"📊 VWAP signal: {'BUY' if is_bull else 'SELL'} | "
            f"dev_sd={dev_sd:.2f} | sl_dist={sl_dist:.2f}pts | "
            f"bars_ext={self._bars_at_extreme} | dev_mom={dev_mom:.2f}"
        )

    # ── Stop/target ───────────────────────────────────────────────────────────

    def get_stop_target_pts(
        self,
        df: pd.DataFrame,
        direction: str,
        entry_price: float,
    ) -> Tuple[Optional[float], Optional[float]]:
        """Stop: bar extreme ± 0.1×ATR. Target: fixed 2R (raw_rr must be ≥ 2.0)."""
        if self._signal_sl_dist <= 0:
            return None, None

        stop_pts = self._signal_sl_dist
        raw_rr   = self._latest_risk_rr

        if raw_rr < 2.0:
            return None, None

        target_pts = 2.0 * stop_pts
        logging.info(
            f"  VWAP stop/target | dir={direction} entry={entry_price:.2f} "
            f"stop={stop_pts:.2f}pts target={target_pts:.2f}pts "
            f"(predicted_rr={raw_rr:.2f} → 2R fixed)"
        )
        return stop_pts, target_pts
