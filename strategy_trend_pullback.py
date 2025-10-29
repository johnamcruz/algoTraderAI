#!/usr/bin/env python3
"""
Trend Pullback Implementation (V7.0)

This strategy uses a Transformer model to find high-probability trend
continuations (pullbacks) based on a 15/40 EMA system.
"""

import pandas as pd
import pandas_ta as ta
import numpy as np
import onnxruntime
import pickle
import os
import logging
from typing import Dict, List, Tuple, Optional
# Assuming strategy_base.py contains the BaseStrategy class
from strategy_base import BaseStrategy


class TrendPullbackStrategy(BaseStrategy):
    """
    Trend Pullback (V7.0) strategy based on EMA 15 / 40 pullbacks.
    """

    def __init__(self, model_path: str, scaler_path: str, contract_symbol: str):
        """
        Initialize Trend Pullback strategy.

        Args:
            model_path: Path to the V7.0 ONNX model (trained with seq_len=40)
            scaler_path: Path to the V7.0 scaler
            contract_symbol: Trading symbol
        """
        super().__init__(model_path, scaler_path, contract_symbol)
        logging.info("Initialized TrendPullbackStrategy (V7.0)")


    def get_feature_columns(self) -> List[str]:
        """Returns the 11 feature columns for the Trend Pullback (V7.0) model."""
        return [
            # --- Price Position Context (3 features) ---
            'price_vs_fast_ema', 'fast_vs_slow_ema', 'pullback_depth',
            # --- Momentum/Vol (5 features) ---
            'adx', 'adx_slope', 'rsi', 'macd_hist_norm', 'chop_factor',
            # --- Volume Confirmation (2 features) ---
            'cmf', 'volume_norm',
            # --- Trend Genesis (1 feature) ---
            'trend_duration'
        ]

    def get_sequence_length(self)-> int:
        """Trend Pullback (V7.0) uses 40 bars."""
        return 40

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all 11 Trend Pullback (V7.0) features.
        Also calculates the 'is_long_pullback' and 'is_short_pullback'
        trigger columns needed for the entry logic.
        """
        # === EMA Parameters ===
        fast_ema = 15
        slow_ema = 40
        trend_ema = 150 # Not used in V7.0 features, but was in the training code

        # === FIX: CLEAN OHLCV DATA FIRST ===
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(method='ffill')
            df[col] = df[col].fillna(0)

        # === Volatility & Base EMAs ===
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['atr'] = df['atr'].replace(0, 1e-6) # Avoid zero division
        df['atr'] = df['atr'].fillna(method='ffill')
        df['atr'] = df['atr'].fillna(1e-6)

        df[f'ema{fast_ema}'] = ta.ema(df['close'], length=fast_ema)
        df[f'ema{slow_ema}'] = ta.ema(df['close'], length=slow_ema)

        # === V7.0 Core Context Features (Normalized) ===
        df['rsi'] = ta.rsi(df['close'], length=14)
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
        df['adx'] = adx_df['ADX_14'] if adx_df is not None and 'ADX_14' in adx_df.columns else 0
        df['adx_slope'] = df['adx'].diff(5) / 5
        
        # Normalized MACD Hist
        macd_df = ta.macd(df['close'], fast=12, slow=26, signal=9)
        df['macd_hist'] = macd_df['MACDh_12_26_9'] if macd_df is not None and 'MACDh_12_26_9' in macd_df.columns else 0
        df['macd_hist_norm'] = df['macd_hist'] / df['atr']
        
        # Normalized Volume
        df['vol_sma_20'] = ta.sma(df['volume'], length=20)
        df['vol_sma_20'] = df['vol_sma_20'].replace(0, 1e-6)
        df['volume_norm'] = df['volume'] / df['vol_sma_20']
        
        # CMF
        df['cmf'] = ta.cmf(df['high'], df['low'], df['close'], df['volume'], length=20)
        
        # Relational Price Position
        df['price_vs_fast_ema'] = (df['close'] - df[f'ema{fast_ema}']) / df['atr']
        df['fast_vs_slow_ema'] = (df[f'ema{fast_ema}'] - df[f'ema{slow_ema}']) / df['atr']

        # === V7.0 NEW "GENESIS" FEATURES ===
        
        # 1. Pullback Depth
        df['pullback_depth'] = (df['close'] - df[f'ema{slow_ema}']) / df['atr']
        
        # 2. Chop Factor (Volatility Compression)
        df['atr_100'] = ta.atr(df['high'], df['low'], df['close'], length=100)
        df['atr_100'] = df['atr_100'].replace(0, 1e-6)
        df['chop_factor'] = df['atr'] / df['atr_100']
        
        # 3. Trend Duration (The key "Genesis" feature)
        trend_direction = (df['fast_vs_slow_ema'] > 0).astype(int) * 2 - 1 
        trend_groups = trend_direction.diff().ne(0).cumsum()
        df['trend_duration_raw'] = trend_groups.groupby(trend_groups).cumcount() + 1
        df['trend_duration'] = (df['trend_duration_raw'].clip(upper=100) / 100) * trend_direction

        # === V7.0 TRIGGER COLUMNS (Not features, but needed for entry) ===
        pullback_range = 0.5 * df['atr'] # Price must be within 0.5 ATR of the Fast EMA

        # Long Pullback Check: Close is near the fast EMA AND fast > slow
        df['is_long_pullback'] = (
            (df['close'] >= df[f'ema{fast_ema}'] - pullback_range) &
            (df['close'] <= df[f'ema{fast_ema}'] + pullback_range) &
            (df[f'ema{fast_ema}'] > df[f'ema{slow_ema}'])
        ).astype(float)

        # Short Pullback Check: Close is near the fast EMA AND fast < slow
        df['is_short_pullback'] = (
            (df['close'] >= df[f'ema{fast_ema}'] - pullback_range) &
            (df['close'] <= df[f'ema{fast_ema}'] + pullback_range) &
            (df[f'ema{fast_ema}'] < df[f'ema{slow_ema}'])
        ).astype(float)

        # --- Final Clean up ---
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill') # Fill gaps
        df = df.fillna(0) # Fill leading NaNs

        return df

    def load_model(self):
        """Load ONNX model for Trend Pullback (V7.0)."""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            self.model = onnxruntime.InferenceSession(self.model_path)
            logging.info(f"✅ Loaded Trend Pullback (V7.0) model: {os.path.basename(self.model_path)}")
        except Exception as e:
            logging.exception(f"❌ Error loading Trend Pullback (V7.0) model: {e}")
            raise

    def load_scaler(self):
        """Load scaler for Trend Pullback (V7.0)."""
        try:
            if not os.path.exists(self.scaler_path):
                raise FileNotFoundError(f"Scaler file not found: {self.scaler_path}")

            with open(self.scaler_path, 'rb') as f:
                scalers = pickle.load(f)

            if self.contract_symbol in scalers:
                self.scaler = scalers[self.contract_symbol]
                logging.info(f"✅ Loaded '{self.contract_symbol}' scaler for Trend Pullback (V7.0)")
            else:
                # Fallback to base ticker (e.g., 'NQ' from 'NQ.F')
                base_ticker = self.contract_symbol.split('.')[0]
                if base_ticker in scalers:
                    self.scaler = scalers[base_ticker]
                    logging.info(f"✅ Loaded base '{base_ticker}' scaler for Trend Pullback (V7.0)")
                else:
                    available = list(scalers.keys())
                    raise ValueError(
                        f"'{self.contract_symbol}' or base '{base_ticker}' scaler not found. "
                        f"Available: {available}"
                    )
        except Exception as e:
            logging.exception(f"❌ Error loading Trend Pullback (V7.0) scaler: {e}")
            raise

    def predict(self, df: pd.DataFrame) -> Tuple[int, float]:
        """Generate prediction using Trend Pullback (V7.0) model."""
        try:
            # Preprocess features
            features = self.preprocess_features(df) # Assumes this exists in BaseStrategy

            # Prepare input for ONNX
            seq_len = self.get_sequence_length() # Will get 40
            if len(features) < seq_len:
                logging.warning(f"⚠️ Not enough data for prediction. Need {seq_len}, have {len(features)}. Returning Hold.")
                return 0, 0.0 # Return Hold (0) with 0 confidence

            # Take last sequence
            X = features[-seq_len:].reshape(1, seq_len, -1).astype(np.float32)

            # Run inference
            input_name = self.model.get_inputs()[0].name
            output_name = self.model.get_outputs()[0].name
            logits = self.model.run([output_name], {input_name: X})[0]

            # Get prediction and confidence
            probs = self._softmax(logits[0])
            prediction = int(np.argmax(probs))
            confidence = float(probs[prediction])

            # prediction 0 = Hold, 1 = SHORT, 2 = LONG
            return prediction, confidence

        except Exception as e:
            logging.exception(f"❌ Prediction error (Trend Pullback): {e}")
            return 0, 0.0 # Return Hold on error

    def should_enter_trade(
        self,
        prediction: int, 
        confidence: float,
        bar: Dict,
        entry_conf: float,
        adx_thresh: float
    ) -> Tuple[bool, Optional[str]]:
        """
        Determine if Trend Pullback (V7.0) entry conditions are met.
        This requires BOTH the model signal and the feature trigger.
        """
        
        # 1. Check Model Confidence
        if confidence < entry_conf:
            return False, None
        
        # 2. Check Feature-Based Trigger
        # These values *must* be present in the 'bar' dict from add_features
        is_long_pullback = bar.get('is_long_pullback', 0.0) == 1.0
        is_short_pullback = bar.get('is_short_pullback', 0.0) == 1.0

        # 3. Check for matching Model Prediction AND Feature Trigger
        
        # V7.0 Mapping: 2 = LONG
        if prediction == 2 and is_long_pullback:
            return True, 'LONG'
        
        # V7.0 Mapping: 1 = SHORT
        elif prediction == 1 and is_short_pullback:
            return True, 'SHORT'

        # All other cases (e.g., pred=0, or pred=2 but bar is not a long pullback)
        return False, None

    @staticmethod
    def _softmax(x):
        """Compute softmax values for array x."""
        if x is None or len(x) == 0:
            return np.array([])
        exp_x = np.exp(x - np.max(x)) # Subtract max for numerical stability
        return exp_x / exp_x.sum()