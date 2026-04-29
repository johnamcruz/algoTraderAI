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
import onnxruntime
from collections import deque
from typing import Dict, List, Tuple, Optional

from strategies.strategy_base import BaseStrategy
from futures_foundation import derive_features, get_model_feature_columns, INSTRUMENT_MAP
from utils.bot_utils import parse_future_symbol, MICRO_TO_MINI_MAP

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ── VWAP parameters (must match vwap_v1.py training / v1.1 model context) ───
VWAP_DEV_THRESH  = 2.0    # live inference trigger: abs(dev_sd) >= 2.0 SDs
SL_ATR_MULT      = 0.1    # stop buffer: bar_extreme ± ATR × 0.1
ATR_PERIOD       = 14     # Wilder's smoothed ATR period
ATR_RANK_WINDOW  = 200    # rolling ATR percentile window (bars)
SD_RANK_WINDOW   = 500    # rolling VWAP-SD percentile window (bars)
SESSION_START    = 7      # ET hour — trading session open
SESSION_END      = 16     # ET hour — trading session close

VWAP_FEATURE_COLS = [
    'vwap_dev_sd',       # 0: signed deviation in intraday SDs (neg=below VWAP=BUY)
    'vwap_dev_pct',      # 1: (close - vwap) / close × 100, signed
    'atr_rank_pct',      # 2: rolling 200-bar ATR percentile [0, 1]
    'htf_vwap_dev_sd',   # 3: 1H-bucket VWAP deviation in SDs, clipped [-5, 5]
    'daily_vwap_trend',  # 4: +1 VWAP rising last 12 bars, -1 falling, 0 flat
    'bars_at_extreme',   # 5: consecutive bars >= DEV_THRESH SDs, capped at 20
    'dev_momentum',      # 6: dev_sd[i] - dev_sd[i-3], clipped [-5, 5]
    'vwap_sd_rank',      # 7: rolling 500-bar VWAP-SD percentile [0, 1]
]


class VWAPReversionStrategyV1(BaseStrategy):
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
        super().__init__(model_path, contract_symbol)
        self._instrument     = self._resolve_instrument(contract_symbol)
        self._min_risk_rr    = min_risk_rr

        # ── Daily VWAP state (resets at midnight ET each trading day) ────────
        self._current_date: Optional[object] = None
        self._cum_tp_vol: float  = 0.0
        self._cum_vol: float     = 0.0
        self._vwap_devs: list    = []   # (tp - vwap) per bar this day
        self._vwap: float        = 0.0
        self._vwap_sd: float     = 0.0

        # ── 1H bucket VWAP (resets each calendar hour) ───────────────────────
        self._htf_hour_key: Optional[tuple] = None
        self._htf_cum_tv: float = 0.0
        self._htf_cum_v: float  = 0.0
        self._htf_devs: list    = []
        self._htf_vwap_dev_sd: float = 0.0

        # ── Wilder ATR (14-period) ────────────────────────────────────────────
        self._atr: float             = 0.0
        self._prev_close: float      = 0.0
        self._atr_initialized: bool  = False

        # ── Rolling percentile windows ────────────────────────────────────────
        self._atr_history: deque = deque(maxlen=ATR_RANK_WINDOW)
        self._sd_history: deque  = deque(maxlen=SD_RANK_WINDOW)

        # ── Lookback deques for momentum / slope features ─────────────────────
        self._recent_dev_sd: deque = deque(maxlen=4)   # dev_sd[i-3] lookback
        self._recent_vwap: deque   = deque(maxlen=13)  # vwap[i-12]  lookback

        # ── Running consecutive-bars counter ──────────────────────────────────
        self._bars_at_extreme: int = 0

        # ── Signal state ──────────────────────────────────────────────────────
        self._is_signal_bar: bool              = False
        self._current_direction: int           = 0   # 0=none, 1=buy, 2=sell
        self._current_vwap_features: np.ndarray = np.zeros(8, dtype=np.float32)
        self._signal_atr: float                = 0.0
        self._signal_sl_dist: float            = 0.0
        self._signal_vwap: float               = 0.0
        self._latest_risk_rr: float            = 0.0
        self._latest_signal_meta: dict         = {}

        self.skip_stats: dict = {'conf_gate': 0, 'rr_gate': 0, 'hold': 0}

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
        return 200

    def get_feature_columns(self) -> List[str]:
        return get_model_feature_columns()

    def is_trading_allowed(self, timestamp: pd.Timestamp) -> bool:
        return True

    def on_trade_exit(self, reason: str):
        self._is_signal_bar    = False
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
        vol   = float(row.get('volume', 0.0) or 0.0)
        ts    = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else None

        self._is_signal_bar = False

        # ── Wilder ATR (14-period) ────────────────────────────────────────────
        if not self._atr_initialized:
            self._atr            = max(high - low, 1e-6)
            self._prev_close     = close
            self._atr_initialized = True
        else:
            tr          = max(high - low,
                              abs(high - self._prev_close),
                              abs(low  - self._prev_close))
            self._atr   = max(((ATR_PERIOD - 1) * self._atr + tr) / ATR_PERIOD, 1e-6)
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

        is_bull = (dev_sd < 0 and close > open_)   # below VWAP, bullish close → BUY
        is_bear = (dev_sd > 0 and close < open_)   # above VWAP, bearish close → SELL
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
            float(dev_sd),                                                # 0: vwap_dev_sd
            float((close - self._vwap) / (close + 1e-6) * 100),          # 1: vwap_dev_pct
            float(atr_rank),                                              # 2: atr_rank_pct
            float(self._htf_vwap_dev_sd),                                 # 3: htf_vwap_dev_sd
            float(vwap_trend),                                            # 4: daily_vwap_trend
            float(self._bars_at_extreme),                                 # 5: bars_at_extreme
            float(dev_mom),                                               # 6: dev_momentum
            float(sd_rank),                                               # 7: vwap_sd_rank
        ], dtype=np.float32)

        logging.info(
            f"📊 VWAP signal: {'BUY' if is_bull else 'SELL'} | "
            f"dev_sd={dev_sd:.2f} | sl_dist={sl_dist:.2f}pts | "
            f"bars_ext={self._bars_at_extreme} | dev_mom={dev_mom:.2f}"
        )

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
            logging.info(f"⏳ VWAP warmup: processing {len(df) - 1} historical bars...")
            self._run_warmup(df)
            logging.info("✅ VWAP warmup done")

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
        expected = {'features', 'strategy_features', 'candle_types',
                    'time_of_day', 'day_of_week', 'instrument_ids', 'session_ids'}
        missing  = expected - {i.name for i in self.model.get_inputs()}
        if missing:
            raise ValueError(
                f"Model incompatible with VWAPReversionStrategyV1. "
                f"Missing inputs: {sorted(missing)}. Use vwap_v11.onnx."
            )
        inputs  = [(i.name, i.shape) for i in self.model.get_inputs()]
        outputs = [(o.name, o.shape) for o in self.model.get_outputs()]
        logging.info(f"  ✅ ONNX loaded: {os.path.basename(self.model_path)}")
        logging.info(f"     Inputs:  {inputs}")
        logging.info(f"     Outputs: {outputs}")

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> Tuple[int, float]:
        """
        Run ONNX inference only on VWAP signal bars.
        Non-signal bars return (0, 0.0) immediately without touching the model.
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
            seq      = feat_arr[-seq_len:].reshape(1, seq_len, -1)

            # VWAP strategy features [1, 8]
            strategy_features = self._current_vwap_features.reshape(1, -1).astype(np.float32)

            # candle_types [1, 96] — 0=doji 1=bull_strong 2=bear_strong 3=bull_pin 4=bear_pin 5=neutral
            if 'candle_type' in df.columns:
                ct = df['candle_type'].fillna(0).values.astype(np.int64)
            else:
                ct = np.zeros(len(df), dtype=np.int64)
            candle_types = ct[-seq_len:].reshape(1, seq_len)

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

            # session_ids [1, 96] — use derive_features sess_id when available
            if 'sess_id' in df.columns:
                sess = df['sess_id'].values.astype(np.int64)
            elif hasattr(df.index, 'hour'):
                h    = df.index.hour
                sess = np.where(h < 7,  0,
                       np.where(h < 9,  1,
                       np.where(h < 13, 2, 3))).astype(np.int64)
            else:
                sess = np.full(len(df), 2, dtype=np.int64)
            session_ids = sess[-seq_len:].reshape(1, seq_len)

            # instrument_ids [1]
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
                'confidence':   round(confidence, 4),
                'risk_rr':      round(self._latest_risk_rr, 4),
                'direction':    'BUY' if self._current_direction == 1 else 'SELL',
                'vwap_dev_sd':  round(float(self._current_vwap_features[0]), 4),
                'dev_momentum': round(float(self._current_vwap_features[6]), 4),
                'sl_dist':      round(self._signal_sl_dist, 4),
            }

            logging.debug(
                f"  VWAP v1 | conf={confidence:.3f} "
                f"dir={'BUY' if self._current_direction == 1 else 'SELL'} "
                f"rr={self._latest_risk_rr:.2f}"
            )
            return self._current_direction, confidence

        except Exception as e:
            logging.exception(f"❌ VWAP Reversion v1 predict error: {e}")
            return 0, 0.0

    def should_enter_trade(
        self,
        prediction: int,
        confidence: float,
        bar: Dict,
        entry_conf: float,
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
            logging.info(f"✅ VWAP Rev BUY  | conf={confidence:.3f} rr={self._latest_risk_rr:.2f}")
            return True, 'LONG'
        if prediction == 2:
            logging.info(f"✅ VWAP Rev SELL | conf={confidence:.3f} rr={self._latest_risk_rr:.2f}")
            return True, 'SHORT'

        self.skip_stats['hold'] += 1
        return False, None

    def get_stop_target_pts(
        self,
        df: pd.DataFrame,
        direction: str,
        entry_price: float,
    ) -> Tuple[Optional[float], Optional[float]]:
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

    def preprocess_features(self, df: pd.DataFrame) -> np.ndarray:
        arr = df[self.get_feature_columns()].values.astype(np.float32)
        return np.nan_to_num(arr, nan=0.0, posinf=5.0, neginf=-5.0)
