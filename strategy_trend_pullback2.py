#!/usr/bin/env python3
"""
Trend Pullback Implementation (V8.0 - Sequence Tagging)

This strategy uses a Transformer model trained via Sequence Tagging to find high-precision
trend continuations (pullbacks) based on a 15/40 EMA system.
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


class TrendPullbackStrategy2(BaseStrategy):
    """
    Trend Pullback (V8.0) strategy based on EMA 15 / 40 pullbacks.
    The prediction logic is updated for Sequence Tagging output (B, T, C).
    """

    def __init__(self, model_path: str, scaler_path: str, contract_symbol: str):
        """
        Initialize Trend Pullback strategy.
        """
        super().__init__(model_path, scaler_path, contract_symbol)
        # Renamed internal logging for V8.0 clarity
        logging.info("Initialized TrendPullbackStrategy (V8.0 Sequence Tagging)")


    def get_feature_columns(self) -> List[str]:
        """Returns the 11 feature columns for the Trend Pullback (V8.0) model."""
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
        """Trend Pullback (V8.0) uses 40 bars."""
        return 40

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all 11 Trend Pullback (V8.0) features.
        (Feature generation logic remains identical to V7.0)
        """
        # === EMA Parameters ===
        fast_ema = 15
        slow_ema = 40

        # === FIX: CLEAN OHLCV DATA FIRST ===
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(method='ffill')
            df[col] = df[col].fillna(0)

        # === Volatility & Base EMAs ===
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['atr'] = df['atr'].replace(0, 1e-6)
        df['atr'] = df['atr'].fillna(method='ffill')
        df['atr'] = df['atr'].fillna(1e-6)

        df[f'ema{fast_ema}'] = ta.ema(df['close'], length=fast_ema)
        df[f'ema{slow_ema}'] = ta.ema(df['close'], length=slow_ema)

        # === V8.0 Core Context Features (Normalized) ===
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

        # === V8.0 NEW "GENESIS" FEATURES ===
        
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

        # === TRIGGER COLUMNS (Needed for should_enter_trade logic) ===
        pullback_range = 0.5 * df['atr'] 

        df['is_long_pullback'] = (
            (df['close'] >= df[f'ema{fast_ema}'] - pullback_range) &
            (df['close'] <= df[f'ema{fast_ema}'] + pullback_range) &
            (df[f'ema{fast_ema}'] > df[f'ema{slow_ema}'])
        ).astype(float)
        df['is_short_pullback'] = (
            (df['close'] >= df[f'ema{fast_ema}'] - pullback_range) &
            (df['close'] <= df[f'ema{fast_ema}'] + pullback_range) &
            (df[f'ema{fast_ema}'] < df[f'ema{slow_ema}'])
        ).astype(float)

        # --- Final Clean up ---
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill')
        df = df.fillna(0)

        return df

    def load_model(self):
        """Load ONNX model for Trend Pullback (V8.0)."""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            self.model = onnxruntime.InferenceSession(self.model_path)
            logging.info(f"✅ Loaded Trend Pullback (V8.0 Sequence Tagging) model: {os.path.basename(self.model_path)}")
        except Exception as e:
            logging.exception(f"❌ Error loading Trend Pullback (V8.0) model: {e}")
            raise

    def load_scaler(self):
        """Load scaler for Trend Pullback (V8.0)."""
        try:
            if not os.path.exists(self.scaler_path):
                raise FileNotFoundError(f"Scaler file not found: {self.scaler_path}")

            with open(self.scaler_path, 'rb') as f:
                scalers = pickle.load(f)

            if self.contract_symbol in scalers:
                self.scaler = scalers[self.contract_symbol]
                logging.info(f"✅ Loaded '{self.contract_symbol}' scaler for Trend Pullback (V8.0)")
            else:
                base_ticker = self.contract_symbol.split('.')[0]
                if base_ticker in scalers:
                    self.scaler = scalers[base_ticker]
                    logging.info(f"✅ Loaded base '{base_ticker}' scaler for Trend Pullback (V8.0)")
                else:
                    available = list(scalers.keys())
                    raise ValueError(
                        f"'{self.contract_symbol}' or base '{base_ticker}' scaler not found. "
                        f"Available: {available}"
                    )
        except Exception as e:
            logging.exception(f"❌ Error loading Trend Pullback (V8.0) scaler: {e}")
            raise

    def predict(self, df: pd.DataFrame) -> Tuple[int, float]:
        """
        MODIFIED for V8.0 Sequence Tagging. 
        Generates prediction using the output of the final (last) bar in the sequence.
        """
        try:
            # Preprocess features (assumes this exists in BaseStrategy)
            features = self.preprocess_features(df) 

            # Prepare input for ONNX
            seq_len = self.get_sequence_length() # 40
            if len(features) < seq_len:
                logging.warning(f"⚠️ Not enough data for prediction. Need {seq_len}, have {len(features)}. Returning Hold.")
                return 0, 0.0 # Return Hold (0) with 0 confidence

            # Take last sequence (Input shape: 1, 40, Num_Features)
            X = features[-seq_len:].reshape(1, seq_len, -1).astype(np.float32)

            # Run inference
            input_name = self.model.get_inputs()[0].name
            output_name = self.model.get_outputs()[0].name
            
            # Logits output shape is (1, 40, 3) for the V8.0 Tagging Model
            logits = self.model.run([output_name], {input_name: X})[0] 

            # --- V8.0 FIX: EXTRACT LOGITS FOR THE LAST BAR ONLY ---
            # We explicitly take the last position of the time dimension (index -1)
            if logits.ndim != 3:
                 # This check handles the error you had previously
                 logging.error(f"Model output shape is incorrect: {logits.shape}. Expected (1, 40, 3).")
                 return 0, 0.0 

            # last_logits shape is (3,)
            last_logits = logits[0][-1] 

            # Get prediction and confidence
            probs = self._softmax(last_logits) # Probs is now size 3 (0, 1, 2)
            prediction = int(np.argmax(probs))
            confidence = float(probs[prediction])

            # prediction 0 = Hold, 1 = SHORT, 2 = LONG
            return prediction, confidence

        except Exception as e:
            # Log the detailed exception for debugging
            logging.exception(f"❌ Prediction error (Trend Pullback V8.0): {e}")
            return 0, 0.0 # Return Hold on error

    def should_enter_trade(
        self,
        prediction: int, 
        confidence: float,
        bar: Dict,
        entry_conf: float,
        adx_thresh: float # Still unused in this V8.0 logic, but remains in signature for compatibility
    ) -> Tuple[bool, Optional[str]]:
        """
        Determine if Trend Pullback (V8.0) entry conditions are met.
        This requires BOTH the model signal and the feature trigger.
        """
        
        # 1. Check Model Confidence
        if confidence < entry_conf:
            return False, None
        
        # 2. Check Feature-Based Trigger (Is price in the pullback zone?)
        is_long_pullback = bar.get('is_long_pullback', 0.0) == 1.0
        is_short_pullback = bar.get('is_short_pullback', 0.0) == 1.0

        # 3. Check for matching Model Prediction AND Feature Trigger
        
        # V8.0 Mapping: 2 = LONG
        if prediction == 2 and is_long_pullback:
            return True, 'LONG'
        
        # V8.0 Mapping: 1 = SHORT
        elif prediction == 1 and is_short_pullback:
            return True, 'SHORT'

        return False, None

    @staticmethod
    def _softmax(x):
        """Compute softmax values for array x."""
        if x is None or len(x) == 0:
            return np.array([])
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()