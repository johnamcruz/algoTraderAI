#!/usr/bin/env python3
"""
FFMStrategyBase — shared infrastructure for all FFM Hybrid Transformer strategies.

All FFM strategies share the same ONNX architecture and inference loop:
  FFMBackbone(256-dim, 67 features × 96 bars) + StrategyProjection(M features)
  → signal_logits [B, 2], risk_predictions [B, 1], confidence [B]

  ONNX inputs (7): features, strategy_features, candle_types,
                   time_of_day, day_of_week, instrument_ids, session_ids

Subclasses must implement three abstract methods:
  _is_ready_to_predict()   — is this bar a signal bar worth running through the model?
  _get_strategy_features() — M-element strategy feature vector for current bar
  _get_signal_direction()  — 0=hold, 1=buy, 2=sell

Optional overrides:
  _after_new_bar()      — post-bar hook (e.g. build CISD feature vector from zone state)
  _build_signal_meta()  — extend the default {confidence, risk_rr} metadata dict
  _get_candle_types()   — default: candle_type column if present, else zeros
  _get_session_ids()    — default: sess_id column (from derive_features), else ET fallback
  get_stop_target_pts() — default: (None, None) to use bot's global stop/target
"""

import logging
import os
from abc import abstractmethod
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import onnxruntime

from strategies.strategy_base import BaseStrategy
from futures_foundation import derive_features, get_model_feature_columns, INSTRUMENT_MAP
from utils.bot_utils import parse_future_symbol, MICRO_TO_MINI_MAP

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class FFMStrategyBase(BaseStrategy):
    """
    Shared infrastructure for all FFM Hybrid Transformer strategies.

    Handles ONNX loading, derive_features() integration, the full 7-input
    inference loop, confidence+RR entry gates, and warmup protocol.
    Each subclass implements only its signal detection and feature construction.
    """

    # Class-level defaults so attributes are always accessible even when
    # __init__ is bypassed (e.g. CISDOTEStrategyV7.__new__ in unit tests).
    _strategy_tag: str    = ""
    _min_risk_rr: float   = 0.0
    _latest_risk_rr: float = 0.0

    def __init__(
        self,
        model_path: str,
        contract_symbol: str,
        min_risk_rr: float = 2.0,
        strategy_tag: str = "",
    ):
        super().__init__(model_path, contract_symbol)
        self._instrument: str          = self._resolve_instrument(contract_symbol)
        self._min_risk_rr: float       = min_risk_rr
        self._latest_risk_rr: float    = 0.0
        self._latest_signal_meta: dict = {}
        self._strategy_tag: str        = strategy_tag
        self.skip_stats: dict          = {'conf_gate': 0, 'rr_gate': 0, 'hold': 0}

    # ── Instrument resolution ─────────────────────────────────────────────────

    @staticmethod
    def _resolve_instrument(contract_symbol: Optional[str]) -> str:
        """Map any contract ID format to the parent instrument symbol (e.g. MNQM26 → NQ)."""
        if not contract_symbol:
            return ''
        if contract_symbol.count('.') >= 3:
            root = contract_symbol.split('.')[-2].upper()
            return MICRO_TO_MINI_MAP.get(root, root)
        return parse_future_symbol(contract_symbol) or contract_symbol.upper()

    # ── BaseStrategy concrete implementations ─────────────────────────────────

    def get_sequence_length(self) -> int:
        return 96

    def get_feature_columns(self) -> List[str]:
        return self._feature_cols if self._feature_cols is not None else get_model_feature_columns()

    def _compute_ffm_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Call derive_features() and merge all 67 FFM feature columns back into df."""
        if isinstance(df.index, pd.DatetimeIndex):
            df_input = df.reset_index().rename(
                columns={df.index.name or 'index': 'datetime'})
        else:
            df_input = df.copy()
            if 'datetime' not in df_input.columns:
                raise ValueError("df must have DatetimeIndex or 'datetime' column")
        feature_df = derive_features(df_input, self._instrument)
        for col in feature_df.columns:
            df[col] = feature_df[col].values
        return df

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        self._instrument = self._resolve_instrument(self.contract_symbol) or self._instrument
        df = df.copy()
        if len(df) < 2:
            return df
        df = self._compute_ffm_features(df)
        if self._bar_count == 0 and len(df) > 1:
            logging.info(
                f"⏳ {self.__class__.__name__} warmup: processing {len(df) - 1} bars...")
            self._run_warmup(df)
            logging.info(f"✅ {self.__class__.__name__} warmup done")
        self._on_new_bar(df, self._bar_count)
        self._after_new_bar(df, self._bar_count)
        self._bar_count += 1
        return df

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        self.model = onnxruntime.InferenceSession(
            self.model_path, providers=['CPUExecutionProvider'])
        expected = {'features', 'strategy_features', 'candle_types',
                    'time_of_day', 'day_of_week', 'instrument_ids', 'session_ids'}
        missing = expected - {i.name for i in self.model.get_inputs()}
        if missing:
            raise ValueError(
                f"Model '{os.path.basename(self.model_path)}' is missing required FFM inputs: "
                f"{sorted(missing)}. Expected: {sorted(expected)}."
            )
        inputs  = [(i.name, i.shape) for i in self.model.get_inputs()]
        outputs = [(o.name, o.shape) for o in self.model.get_outputs()]
        logging.info(f"  ✅ ONNX loaded: {os.path.basename(self.model_path)}")
        logging.info(f"     Inputs:  {inputs}")
        logging.info(f"     Outputs: {outputs}")
        self._load_feature_cols_from_metadata()

    def predict(self, df: pd.DataFrame) -> Tuple[int, float]:
        try:
            seq_len      = self.get_sequence_length()
            feature_cols = self.get_feature_columns()

            if df.empty or len(df) < seq_len:
                return 0, 0.0
            if not self._is_ready_to_predict():
                return 0, 0.0

            missing = [c for c in feature_cols if c not in df.columns]
            if missing:
                logging.warning(f"⚠️ Missing FFM features: {missing[:5]}...")
                return 0, 0.0

            feat_arr = df[feature_cols].values.astype(np.float32)
            feat_arr = np.nan_to_num(feat_arr, nan=0.0, posinf=5.0, neginf=-5.0)
            feat_arr = np.clip(feat_arr, -10.0, 10.0)
            seq      = feat_arr[-seq_len:].reshape(1, seq_len, -1)

            strategy_features = self._get_strategy_features().reshape(1, -1).astype(np.float32)
            candle_types  = self._get_candle_types(df, seq_len)
            time_of_day   = self._get_time_of_day(df, seq_len)
            day_of_week   = self._get_day_of_week(df, seq_len)
            session_ids   = self._get_session_ids(df, seq_len)

            self._instrument = self._resolve_instrument(self.contract_symbol) or self._instrument
            instrument_ids   = np.array([INSTRUMENT_MAP.get(self._instrument, 0)], dtype=np.int64)

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
            self._latest_signal_meta = self._build_signal_meta(confidence)
            prediction = self._get_signal_direction()

            logging.debug(
                f"  {self._strategy_tag} | conf={confidence:.3f} "
                f"dir={'BUY' if prediction == 1 else 'SELL' if prediction == 2 else 'NONE'}"
            )
            return prediction, confidence

        except Exception as e:
            logging.exception(f"❌ {self.__class__.__name__} predict error: {e}")
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
            logging.info(
                f"✅ {self._strategy_tag} BUY  | conf={confidence:.3f} rr={self._latest_risk_rr:.2f}")
            return True, 'LONG'
        if prediction == 2:
            logging.info(
                f"✅ {self._strategy_tag} SELL | conf={confidence:.3f} rr={self._latest_risk_rr:.2f}")
            return True, 'SHORT'
        self.skip_stats['hold'] += 1
        return False, None

    def preprocess_features(self, df: pd.DataFrame) -> np.ndarray:
        arr = df[self.get_feature_columns()].values.astype(np.float32)
        return np.nan_to_num(arr, nan=0.0, posinf=5.0, neginf=-5.0)

    def get_stop_target_pts(
        self,
        df: pd.DataFrame,
        direction: str,
        entry_price: float,
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Return strategy-derived (stop_pts, target_pts), or (None, None) to use
        the bot's global --stop_pts / --target_pts.

        Common RR-tier pattern used by zone/ATR-based strategies:
          raw_rr = self._latest_risk_rr
          if raw_rr < 2.0: return None, None
          rr = int(raw_rr)            # floor to integer tier (2R, 3R, 4R …)
          target_pts = stop_pts * rr
        """
        return None, None

    # ── Abstract hooks — subclasses must implement ────────────────────────────

    @abstractmethod
    def _is_ready_to_predict(self) -> bool:
        """Return True only on bars where this strategy has a signal to run through the model."""

    @abstractmethod
    def _get_strategy_features(self) -> np.ndarray:
        """Return the M-element strategy feature vector for the current bar."""

    @abstractmethod
    def _get_signal_direction(self) -> int:
        """Return 0=hold, 1=buy, 2=sell for the current bar."""

    # ── Optional hooks — override to customise ────────────────────────────────

    def _after_new_bar(self, df: pd.DataFrame, bar_idx: int) -> None:
        """Called after _on_new_bar in add_features. Override for post-bar computation."""
        pass

    def _build_signal_meta(self, confidence: float) -> dict:
        return {
            'confidence': round(confidence, 4),
            'risk_rr':    round(self._latest_risk_rr, 4),
        }

    # ── ONNX input builders — override per strategy when training differs ─────

    def _get_candle_types(self, df: pd.DataFrame, seq_len: int) -> np.ndarray:
        """Default: from candle_type column if present, else zeros."""
        if 'candle_type' in df.columns:
            ct = df['candle_type'].fillna(0).values.astype(np.int64)
        else:
            ct = np.zeros(len(df), dtype=np.int64)
        return ct[-seq_len:].reshape(1, seq_len)

    def _get_session_ids(self, df: pd.DataFrame, seq_len: int) -> np.ndarray:
        """Default: sess_id column provided by derive_features; ET-bucket fallback."""
        if 'sess_id' in df.columns:
            sess = df['sess_id'].values.astype(np.int64)
        elif hasattr(df.index, 'hour'):
            h    = df.index.hour
            sess = np.where(h < 3, 0,
                   np.where(h < 8, 1,
                   np.where(h < 12, 2, 3))).astype(np.int64)
        else:
            sess = np.full(len(df), 2, dtype=np.int64)
        return sess[-seq_len:].reshape(1, seq_len)

    def _get_time_of_day(self, df: pd.DataFrame, seq_len: int) -> np.ndarray:
        if 'sess_time_of_day' in df.columns:
            tod = df['sess_time_of_day'].values.astype(np.float32)
        elif hasattr(df.index, 'hour'):
            tod = ((df.index.hour * 60 + df.index.minute) / 1440.0).astype(np.float32)
        else:
            tod = np.zeros(len(df), dtype=np.float32)
        return tod[-seq_len:].reshape(1, seq_len)

    def _get_day_of_week(self, df: pd.DataFrame, seq_len: int) -> np.ndarray:
        if 'tmp_day_of_week' in df.columns:
            dow = df['tmp_day_of_week'].values.astype(np.int64)
        elif hasattr(df.index, 'dayofweek'):
            dow = df.index.dayofweek.values.astype(np.int64)
        else:
            dow = np.zeros(len(df), dtype=np.int64)
        return dow[-seq_len:].reshape(1, seq_len)
