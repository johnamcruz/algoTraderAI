#!/usr/bin/env python3
"""
FVG Reversal Strategy Implementation (V1 - Transformer)

This strategy uses a Transformer model trained on Fair Value Gap (FVG)
events to predict high-probability reversals.
"""

import pandas as pd
import pandas_ta as ta
import numpy as np
import onnxruntime
import pickle
import os
import logging
import math
from typing import Dict, List, Tuple, Optional
# Assuming strategy_base.py contains the BaseStrategy class
from strategy_base import BaseStrategy

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class FVGReversal3minStrategy(BaseStrategy):
    """
    FVG Reversal V1.0 (Transformer) trading strategy.
    Optimized for low-drawdown, fixed-point R:R execution.
    """

    def __init__(self, model_path: str, scaler_path: str, contract_symbol: str):
        """
        Initialize FVG Reversal V1.0 strategy.
        Note: The strategy no longer uses a lookback parameter.
        """
        super().__init__(model_path, scaler_path, contract_symbol)
        
    def get_feature_columns(self) -> List[str]:
        """
        UPDATED: Returns the feature columns for the FVG V1 Transformer model.
        (Matches the Feature List used in Script 1 and 2)
        """
        return [
            # 1. Technical (Signal)
            'fvg_signal',           # -1 (Bear FVG), 0 (None), 1 (Bull FVG)
            'fvg_age',              # How many bars old is the FVG?
            'dist_to_fvg_mid',      # Distance to 50% "consequent encroachment"
            'is_in_discount',       # Price is below 50% of recent range
            'is_in_premium',        # Price is above 50% of recent range
            'market_structure',     # -1 (Down), 1 (Up) based on EMA 200

            # 2. Technical (Context)
            'price_vs_ema200',      # Distance from 200 EMA (Trend)
            'adx',                  # Trend Strength
            'price_vel_20',         # Price Velocity / Momentum
            'body_size',            # Candle body size
            'wick_ratio',           # Candle wick ratio
            'atr',                  # Volatility
            'macro_trend_slope',    # 15-min macro trend

            # 3. Contextual (Time)
            'hour_sin',             # Time of day (cyclical)
            'hour_cos',             # Time of day (cyclical)
            'day_of_week_encoded',  # Day of week

            # 4. Confirmation (Volume/Momentum)
            'mfi',                  # Money Flow Index
            'volume_roc',           # Volume Rate-of-Change
            'stoch_rsi',            # StochRSI
        ]

    # Sequence length is kept at 120 (a typical value in successful Transformer training)
    def get_sequence_length(self) -> int:
        """FVG V1.0 (Transformer) uses 120 bars."""
        return 120

    # =========================================================
    # FVG FEATURE ENGINEERING (REPLACED PIVOT LOGIC)
    # =========================================================
    
    def _find_fvgs(self, high: np.ndarray, low: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized function to find FVG zones (copied from Script 1)."""
        high_m2 = np.roll(high, 2); low_m2 = np.roll(low, 2)

        # Bullish FVG: low[-2] > high[0] (Gap Down)
        bull_fvg = low_m2 > high
        bull_top = low_m2; bull_bottom = high

        # Bearish FVG: high[-2] < low[0] (Gap Up)
        bear_fvg = high_m2 < low
        bear_top = low; bear_bottom = high_m2

        fvg_signal = np.zeros_like(high, dtype=np.int8); fvg_signal[bull_fvg] = 1; fvg_signal[bear_fvg] = -1
        fvg_top = np.where(bull_fvg, bull_top, np.where(bear_fvg, bear_top, np.nan))
        fvg_bottom = np.where(bull_fvg, bull_bottom, np.where(bear_fvg, bear_bottom, np.nan))
        fvg_mid = (fvg_top + fvg_bottom) / 2.0

        fvg_signal[:2] = 0; fvg_top[:2] = np.nan; fvg_mid[:2] = np.nan; fvg_bottom[:2] = np.nan
        return fvg_signal, fvg_top, fvg_mid, fvg_bottom


    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        UPDATED: Calculate FVG V1.0 features.
        """        
        logging.debug(f"Adding FVG V1 features. Input shape: {df.shape}")
        
        df = df.copy()

        # === 0. CLEAN OHLCV DATA ===
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(method='ffill').fillna(0)             

        # === 1. BASE INDICATORS ===
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14).replace(0, 1e-6).fillna(method='ffill').fillna(1e-6)
        df['ema200'] = ta.ema(df['close'], length=200).fillna(method='ffill').fillna(0)

        adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
        df['adx'] = adx_df['ADX_14'] if adx_df is not None and 'ADX_14' in adx_df.columns else 0
        df['adx'] = df['adx'].fillna(method='ffill').fillna(0)

        # === 2. FVG SIGNAL FEATURES ===
        fvg_signal, fvg_top, fvg_mid, fvg_bottom = self._find_fvgs(
            df['high'].values, df['low'].values
        )
        df['fvg_signal_raw'] = fvg_signal
        df['fvg_top_raw'] = fvg_top
        df['fvg_mid_raw'] = fvg_mid
        df['fvg_bottom_raw'] = fvg_bottom

        # Forward fill FVG zones to find the *nearest unmitigated FVG* (limit 100 bars)
        df_fvg = df[['fvg_signal_raw', 'fvg_top_raw', 'fvg_mid_raw', 'fvg_bottom_raw']].replace(0, np.nan)
        df_fvg = df_fvg.ffill(limit=100).fillna(0)
        
        df['fvg_signal'] = df_fvg['fvg_signal_raw']
        df['fvg_top'] = df_fvg['fvg_top_raw']
        df['fvg_mid'] = df_fvg['fvg_mid_raw']
        
        # Calculate FVG age 
        df['fvg_age'] = df.groupby(
            (df['fvg_signal_raw'] != 0).cumsum()
        ).cumcount()
        
        # Calculate distance to FVG mid-point (consequent encroachment)
        df['dist_to_fvg_mid'] = (df['fvg_mid'] - df['close']) / df['atr'] # Reversed sign for consistency
        df.loc[df['fvg_signal'] == 0, 'dist_to_fvg_mid'] = 0

        # Calculate Premium/Discount (using 50-bar swing range)
        swing_range_high = df['high'].rolling(50).max()
        swing_range_low = df['low'].rolling(50).min()
        swing_range_mid = (swing_range_high + swing_range_low) / 2.0
        
        df['is_in_discount'] = np.where(df['close'] < swing_range_mid, 1, 0)
        df['is_in_premium'] = np.where(df['close'] > swing_range_mid, 1, 0)
        df['market_structure'] = np.where(df['close'] > df['ema200'], 1, -1)


        # === 3. CONTEXT & CONFIRMATION FEATURES ===
        df['price_vs_ema200'] = (df['close'] - df['ema200']) / df['atr']
        df['price_vel_20'] = df['close'].diff(20) / df['atr']
        
        # Candlestick
        df['body_size'] = abs(df['close'] - df['open']) / df['atr']
        df['wick_ratio'] = (df['high'] - df['low']) / df['atr'] 

        # Volume/Momentum
        mfi_df = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=14)
        df['mfi'] = mfi_df.fillna(50)
        df['volume_roc'] = ta.roc(df['volume'], length=5).fillna(0)
        stochrsi_df = ta.stochrsi(df['close'], length=14)
        df['stoch_rsi'] = stochrsi_df['STOCHRSIk_14_14_3_3'].fillna(0)

        # Time Context
        hours_in_day = 24
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / hours_in_day)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / hours_in_day)
        df['day_of_week_encoded'] = df.index.dayofweek / 6.0

        # Macro Trend (Placeholder calculation, assumes 15min data is handled externally or derived)
        # For simplicity and compliance, we will mimic the macro trend feature calculation using EMA slope
        df['macro_trend_slope'] = df['ema200'].diff(100) / (df['atr'] * 100) # Proxy for 15m trend from 3m data


        # === FINAL CLEANUP ===
        all_feature_cols = self.get_feature_columns()
        for col in all_feature_cols:
            if col not in df.columns: df[col] = 0.0
                
        df[all_feature_cols] = df[all_feature_cols].replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)

        logging.debug(f"FVG features added. Shape after features: {df.shape}")
        return df    

    # =========================================================
    # LIVE EXECUTION FUNCTIONS
    # =========================================================

    def load_model(self):
        """Load ONNX model for FVG Reversal."""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            # FVG model uses ONNX on CPU
            self.model = onnxruntime.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
            logging.info(f"✅ Loaded FVG Reversal V1 model: {os.path.basename(self.model_path)}")
        except Exception as e:
            logging.exception(f"❌ Error loading FVG Reversal V1 model: {e}")
            raise

    def load_scaler(self):
        """Load scaler for FVG Reversal."""
        try:
            if not os.path.exists(self.scaler_path):
                raise FileNotFoundError(f"Scaler file not found: {self.scaler_path}")

            with open(self.scaler_path, 'rb') as f:
                scalers = pickle.load(f)

            # FVG V1 scalers are keyed by base symbol (e.g., 'NQ')
            base_symbol = self.contract_symbol.split('.')[0][:2] 
            
            if base_symbol in scalers:
                self.scaler = scalers[base_symbol]
                logging.info(f"✅ Loaded '{base_symbol}' scaler for FVG Reversal V1")
            else:
                available = list(scalers.keys())
                raise ValueError(
                    f"'{base_symbol}' scaler not found. "
                    f"Available: {available}"
                )
        except Exception as e:
            logging.exception(f"❌ Error loading FVG Reversal V1 scaler: {e}")
            raise

    def predict(self, df: pd.DataFrame) -> Tuple[int, float]:
        """
        Generate prediction using FVG V1 Transformer model.
        """
        try:
            features = self.preprocess_features(df) 

            seq_len = self.get_sequence_length() # 120
            if len(features) < seq_len:
                logging.warning(f"⚠️ Not enough data for prediction. Need {seq_len}, have {len(features)}. Returning Hold.")
                return 0, 0.0

            X = features[-seq_len:].reshape(1, seq_len, -1).astype(np.float32)

            input_name = self.model.get_inputs()[0].name
            output_name = self.model.get_outputs()[0].name
            logits_sequence = self.model.run([output_name], {input_name: X})[0] # Shape: (1, 120, 3)

            # Get logits from the LAST time step
            last_logits = logits_sequence[0, -1, :] # Shape: (3,)

            # Get prediction and confidence
            probs = self._softmax(last_logits)
            prediction = int(np.argmax(probs))
            confidence = float(probs[prediction])

            return prediction, confidence

        except Exception as e:
            logging.exception(f"❌ Prediction error (FVG Reversal V1): {e}")
            return 0, 0.0 # Return Hold on error

    def should_enter_trade(
        self,
        prediction: int,
        confidence: float,
        bar: Dict,
        entry_conf: float,
        adx_thresh: float # PARAMETER RETAINED FOR COMPLIANCE
    ) -> Tuple[bool, Optional[str]]:
        """
        Determine if FVG Reversal V1 entry conditions are met.
        NOTE: adx_thresh is ignored as the FVG model has learned this context internally.
        """
        
        # Check confidence threshold (Model is trained with high confidence in mind)
        if confidence < entry_conf:
            return False, None
        
        # FVG V1 model uses: 1=BUY (Long), 2=SELL (Short)
        if prediction == 1:
            return True, 'LONG'
        elif prediction == 2:
            return True, 'SHORT'

        # Prediction is 0 (Hold)
        return False, None

    @staticmethod
    def _softmax(x):
        """Compute softmax values for array x."""
        if x is None or len(x) == 0:
            return np.array([])
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()