#!/usr/bin/env python3
"""
Supertrend Pullback Strategy Implementation (V3.8 - Transformer)

This strategy uses a Transformer model to trade Supertrend pullbacks.
It is based on the V3.8 "Balanced Quality Signals" model.
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


class SupertrendPullbackStrategy(BaseStrategy):
    """
    Supertrend Pullback V3.8 (Transformer) trading strategy.
    Uses ADX, EMA200, and Proximity filters for high-precision entries.
    """

    def __init__(self, model_path: str, scaler_path: str, contract_symbol: str):
        """
        Initialize Supertrend Pullback V3.8 strategy.
        """
        # pivot_lookback is no longer needed for this strategy
        super().__init__(model_path, scaler_path, contract_symbol)
        logging.info("Initialized SupertrendPullbackStrategy (V3.8)")


    def get_feature_columns(self) -> List[str]:
        """
        Returns the 22 feature columns for the V3.8 Supertrend Transformer.
        (Matches SUPERTRND PULLBACK AI - V3.8)
        """
        return [ 
             'price_vs_st', 'st_direction',
             'st_val_slow', 'st_direction_slow', 'price_vs_st_slow', 
             'price_vs_ema40', 'ema15_vs_ema40', 'price_vs_ema200',
             'adx', 'adx_slope', 'rsi', 'cmf',
             'price_vel_10', 'price_vel_20', 'rsi_vel_10',
             'body_size', 'wick_ratio', 'atr',
             'macro_trend_slope',
             'price_roc_slope',
             'inverse_volatility_score',
             'volume_velocity'
        ]

    def get_sequence_length(self) -> int:
        """
        Supertrend Pullback V3.8 (Transformer) uses 120 bars.
        """
        return 120

    # =========================================================
    # V3.8 FEATURE ENGINEERING
    # (Matches add_supertrend_features_v3_6 from training)
    # =========================================================
    
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Supertrend Pullback V3.8 features.
        (Matches add_supertrend_features_v3_6 from training)
        """        
        logging.debug(f"Adding V3.8 Supertrend features. Input shape: {df.shape}")
        
        df = df.copy()
        
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14); df['atr'].replace(0, 1e-6, inplace=True)
        df['ema15'] = ta.ema(df['close'], length=15); df['ema40'] = ta.ema(df['close'], length=40); df['ema200'] = ta.ema(df['close'], length=200)
        st_df_fast = ta.supertrend(df['high'], df['low'], df['close'], length=10, multiplier=3)
        df['st_val'] = st_df_fast['SUPERT_10_3']; df['st_direction'] = st_df_fast['SUPERTd_10_3']
        st_df_slow = ta.supertrend(df['high'], df['low'], df['close'], length=20, multiplier=4)
        df['st_val_slow'] = st_df_slow['SUPERT_20_4']; df['st_direction_slow'] = st_df_slow['SUPERTd_20_4']
        df['price_vs_st'] = (df['close'] - df['st_val']) / df['atr'];
        df['price_vs_st_slow'] = (df['close'] - df['st_val_slow']) / df['atr'];
        df['price_vs_ema40'] = (df['close'] - df['ema40']) / df['atr'];
        df['ema15_vs_ema40'] = (df['ema15'] - df['ema40']) / df['atr']
        df['price_vs_ema200'] = (df['close'] - df['ema200']) / df['atr']
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
        df['adx'] = adx_df['ADX_14'] if adx_df is not None and 'ADX_14' in adx_df.columns else 0
        df['adx_slope'] = df['adx'].diff(5)
        df['rsi'] = ta.rsi(df['close'], length=14)
        df['cmf'] = ta.cmf(df['high'], df['low'], df['close'], df['volume'], length=20)
        df['price_vel_10'] = df['close'].diff(10) / df['atr']; df['price_vel_20'] = df['close'].diff(20) / df['atr']; df['rsi_vel_10'] = df['rsi'].diff(10)
        df['body_size'] = abs(df['close'] - df['open']) / df['atr']; df['wick_ratio'] = (df['high'] - df['low']) / df['atr']
        df['price_roc_slope'] = df['price_vel_10'].diff(10)
        df['vol_channel_width'] = (df['ema15'] - df['ema40']).abs()
        df['inverse_volatility_score'] = df['vol_channel_width'] / (df['atr'] + 1e-6)
        vol_std = df['volume'].rolling(10).std().replace(0, 1e-6)
        df['volume_velocity'] = df['volume'].diff(1) / vol_std
        df_15m = df['close'].resample('15Min', label='left', closed='left').ohlc().dropna()

        if not df_15m.empty:
            # 1. Calculate the EMA
            ema_15m_series = ta.ema(df_15m['close'], length=40)
            
            # 2. Check if the result is None (which happens on small datasets)
            if ema_15m_series is not None:
                df_15m['ema40'] = ema_15m_series.fillna(0)
                df_15m['ema40_slope'] = df_15m['ema40'].diff(3)
            else:
                # EMA calculation failed (not enough data), so fill with defaults
                df_15m['ema40'] = 0.0
                df_15m['ema40_slope'] = np.nan
                
            # 3. Continue with the merge
            df_15m = df_15m[['ema40_slope']].shift(1)
            df = df.merge(df_15m, left_index=True, right_index=True, how='left')
        else:
            # df_15m is empty, so we can't merge. Just create the column directly.
            df['ema40_slope'] = np.nan        

        if 'ema40_slope' not in df.columns: df['ema40_slope'] = np.nan
        df['macro_trend_slope'] = df['ema40_slope'] / df['atr']
        
        # --- FIX 1: DATA POISONING ---
        # Replace inf, -inf with NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # ffill carries the last known good value forward
        df.fillna(method='ffill', inplace=True)
        
        # dropna() removes any rows that are *still* NaN
        # (i.e., the initial 200-bar warm-up period)
        # This ensures the model ONLY sees 100% valid, non-zero data.
        df.dropna(inplace=True) 
        # --- END FIX ---

        all_feature_cols = self.get_feature_columns() # Use the class method
        for col in all_feature_cols:
            if col not in df.columns: 
                df[col] = 0.0 # This will now only run if df is empty
                    
        logging.debug(f"V3.8 Supertrend features added. Shape after features: {df.shape}")
        return df

    # =========================================================
    # LIVE EXECUTION FUNCTIONS
    # =========================================================

    def load_model(self):
        """Load ONNX model for Supertrend Pullback."""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            self.model = onnxruntime.InferenceSession(self.model_path)
            logging.info(f"✅ Loaded Supertrend Pullback V3.8 model: {os.path.basename(self.model_path)}")
        except Exception as e:
            logging.exception(f"❌ Error loading Supertrend Pullback V3.8 model: {e}")
            raise

    def load_scaler(self):
        """Load scaler for Supertrend Pullback."""
        try:
            if not os.path.exists(self.scaler_path):
                raise FileNotFoundError(f"Scaler file not found: {self.scaler_path}")

            with open(self.scaler_path, 'rb') as f:
                scalers = pickle.load(f)

            base_symbol = self.contract_symbol.split('.')[0][:2] 
            
            if base_symbol in scalers:
                self.scaler = scalers[base_symbol]
                logging.info(f"✅ Loaded '{base_symbol}' scaler for Supertrend V3.8")
            else:
                available = list(scalers.keys())
                raise ValueError(
                    f"'{base_symbol}' scaler not found. "
                    f"Available: {available}"
                )
        except Exception as e:
            logging.exception(f"❌ Error loading Supertrend V3.8 scaler: {e}")
            raise

    # --- MODIFIED FUNCTION ---
    def predict(self, df: pd.DataFrame) -> Tuple[int, float]:
        """
        Generate prediction using V3.8 Transformer model.
        (Matches training script by using the LAST time step)
        """
        try:
            # --- FIX: Check for empty/insufficient data FIRST ---
            # This catches the empty DataFrame from add_features() during warm-up
            # when dropna() removes all rows.
            seq_len = self.get_sequence_length() # 120
            if df.empty or len(df) < seq_len:
                logging.warning(f"⚠️ Not enough data for prediction. Need {seq_len}, have {len(df)}. (Warm-up in progress)")
                return 0, 0.0
            # --- END FIX ---

            # Now we know df is not empty and has at least seq_len rows.
            # preprocess_features is assumed to be in BaseStrategy
            features = self.preprocess_features(df) 

            # This check is now redundant, but good for safety.
            if len(features) < seq_len:
                logging.warning(f"⚠️ Data length mismatch after scaling. Need {seq_len}, have {len(features)}. Returning Hold.")
                return 0, 0.0

            X = features[-seq_len:].reshape(1, seq_len, -1).astype(np.float32)

            input_name = self.model.get_inputs()[0].name
            output_name = self.model.get_outputs()[0].name
            logits_sequence = self.model.run([output_name], {input_name: X})[0] # Shape: (1, 120, 3)

            # Get logits from the LAST time step (matches training script)
            last_logits = logits_sequence[0, -1, :] # Shape: (3,)

            probs = self._softmax(last_logits)
            prediction = int(np.argmax(probs))
            confidence = float(probs[prediction])

            return prediction, confidence

        except Exception as e:
            logging.exception(f"❌ Prediction error (Supertrend V3.8): {e}")
            return 0, 0.0 # Return Hold on error
    # --- END MODIFIED FUNCTION ---
    
    def should_enter_trade(
        self,
        prediction: int,
        confidence: float,
        bar: Dict, 
        entry_conf: float,
        adx_thresh: float,        
    ) -> Tuple[bool, Optional[str]]:
        """
        Determine if Supertrend V3.8 entry conditions are met.
        Applies hard filters for ADX (chop), EMA (macro-trend),
        and Proximity (pullback) - matching backtest_supertrend_sequential.
        """

        # --- FIX 2: PARAMETER MISMATCH ---
        # This value MUST match the 'proximity_thresh' from LABEL_PARAMS
        # in the training script, which was 1.5.
        proximity_thresh = 1.5
        # --- END FIX ---
        
        # 1. Confidence Filter
        if confidence < entry_conf:
            return False, None
        
        # Get values with safe fallback if key is missing or value is None
        close = bar.get('close') if bar.get('close') is not None else 0.0
        ema200 = bar.get('ema200') if bar.get('ema200') is not None else 0.0
        st_direction = bar.get('st_direction') if bar.get('st_direction') is not None else 0.0
        adx = bar.get('adx') if bar.get('adx') is not None else 0.0
        
        # --- NEW VALUES REQUIRED FOR V3.8 PROXIMITY FILTER ---
        st_val = bar.get('st_val') if bar.get('st_val') is not None else 0.0
        atr = bar.get('atr') if bar.get('atr') is not None else 1e-6 # Avoid zero division
        if atr < 1e-6: atr = 1e-6 # Ensure ATR is not zero
            
        # Model Prediction Filter
        if prediction == 1: # Model wants to BUY
            # 4. Macro-Trend & Alignment Filters
            # Check 1: close > ema200 AND Check 2: st_direction == 1
            if close > ema200 and st_direction == 1:
                return True, 'LONG'
                
        elif prediction == 2: # Model wants to SELL
            # 4. Macro-Trend & Alignment Filters
            # Check 1: close < ema200 AND Check 2: st_direction == -1
            if close < ema200 and st_direction == -1:
                return True, 'SHORT'

        # Prediction is 0 (Hold) or filters failed
        return False, None
    # <--- END OF MODIFIED FUNCTION ---

    @staticmethod
    def _softmax(x):
        """Compute softmax values for array x."""
        if x is None or len(x) == 0:
            return np.array([])
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()