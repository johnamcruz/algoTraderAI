#!/usr/bin/env python3
"""
Supertrend Pullback Strategy Implementation (V3.10 - Transformer)
This strategy is based on the final, honest training configuration.
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
    Supertrend Pullback V3.10 (Transformer) trading strategy.
    Uses "Pure AI Mode" with clean, honest features.
    """

    def __init__(self, model_path: str, scaler_path: str, contract_symbol: str):
        """
        Initialize SupertrendPullbackStrategy (V3.10)
        """
        super().__init__(model_path, scaler_path, contract_symbol)
        logging.info("Initialized SupertrendPullbackStrategy (V3.10 - Pure AI)")
        # Store long-term MTF history (critical for causal processing)
        self.mtf_history: Dict[str, pd.DataFrame] = {}


    def get_feature_columns(self) -> List[str]:
        """
        Returns the 25 feature columns for the V3.10 Supertrend Transformer.
        """
        return [ 
             'price_vs_st', 'st_direction',
             'st_val_slow', 'st_direction_slow', 'price_vs_st_slow', 
             'price_vs_ema40', 'ema15_vs_ema40', 'price_vs_ema200',
             'adx', 'adx_slope', 'rsi', 'cmf',
             'price_vel_10', 'price_vel_20', 'rsi_vel_10',
             'body_size', 'wick_ratio', 'atr',
             'macro_trend_slope', # This is the MTF feature
             'price_roc_slope',
             'inverse_volatility_score',
             'volume_velocity',
             'st_slope_long',
             'dist_to_ema200',
             'adx_acceleration_5'
        ]

    def get_sequence_length(self) -> int:
        """
        Supertrend Pullback V3.10 (Transformer) uses 120 bars.
        """
        return 120

    # =========================================================
    # V3.10 FEATURE ENGINEERING (Self-Contained & Honest)
    # =========================================================
    
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Supertrend Pullback V3.10 features.
        NOTE: This now generates the 15-min feature causally using df's history.
        """        
        logging.debug(f"Adding V3.10 Supertrend features. Input shape: {df.shape}")
        
        df = df.copy()
        
        # --- 1. Basic Indicators (3-min TF) ---
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14).replace(0, 1e-6)
        df['ema15'] = ta.ema(df['close'], length=15); df['ema40'] = ta.ema(df['close'], length=40); df['ema200'] = ta.ema(df['close'], length=200)
        st_df_fast = ta.supertrend(df['high'], df['low'], df['close'], length=10, multiplier=3)
        df['st_val'] = st_df_fast['SUPERT_10_3']; df['st_direction'] = st_df_fast['SUPERTd_10_3']
        st_df_slow = ta.supertrend(df['high'], df['low'], df['close'], length=20, multiplier=4)
        df['st_val_slow'] = st_df_slow['SUPERT_20_4']; df['st_direction_slow'] = st_df_slow['SUPERTd_20_4']
        
        # --- 2. Price/Momentum Features (3-min TF) ---
        df['price_vs_st'] = (df['close'] - df['st_val']) / df['atr'];
        df['price_vs_st_slow'] = (df['close'] - df['st_val_slow']) / df['atr'];
        df['price_vs_ema40'] = (df['close'] - df['ema40']) / df['atr'];
        df['ema15_vs_ema40'] = (df['ema15'] - df['ema40']) / df['atr']
        df['price_vs_ema200'] = (df['close'] - df['ema200']) / df['atr']
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
        df['adx'] = adx_df['ADX_14'] if adx_df is not None and 'ADX_14' in adx_df.columns else 0
        df['adx_slope'] = df['adx'].diff(5)
        df['st_slope_long'] = df['st_val_slow'].diff(50) / (df['atr'] * 50)                
        df['dist_to_ema200'] = abs(df['close'] - df['ema200']) / df['atr']        
        df['adx_acceleration_5'] = df['adx_slope'].diff(5)
        df['rsi'] = ta.rsi(df['close'], length=14)
        df['cmf'] = ta.cmf(df['high'], df['low'], df['close'], df['volume'], length=20)
        df['price_vel_10'] = df['close'].diff(10) / df['atr']; df['price_vel_20'] = df['close'].diff(20) / df['atr']; df['rsi_vel_10'] = df['rsi'].diff(10)
        df['body_size'] = abs(df['close'] - df['open']) / df['atr']; df['wick_ratio'] = (df['high'] - df['low']) / df['atr']
        df['price_roc_slope'] = df['price_vel_10'].diff(10)
        df['vol_channel_width'] = (df['ema15'] - df['ema40']).abs()
        df['inverse_volatility_score'] = df['vol_channel_width'] / (df['atr'] + 1e-6)
        vol_std = df['volume'].rolling(10).std().replace(0, 1e-6); df['volume_velocity'] = df['volume'].diff(1) / vol_std
        
        # --- 3. MTF Feature (Self-Generation - HONEST) ---
        
        # Use the entire current history (df) for causal resampling
        df_15m = df['close'].resample('15Min', label='left', closed='left').ohlc().dropna()

        # Calculate MTF indicators on the 15-min frame
        if not df_15m.empty and 'close' in df_15m.columns:
            ema_15m_series = ta.ema(df_15m['close'], length=40)
            df_15m_features = pd.DataFrame(index=df_15m.index)
            df_15m_features['ema40_slope'] = ema_15m_series.diff(3)
            
            # CRITICAL: Shift by 1 period (15min) to ensure honesty
            df_15m_features = df_15m_features.shift(1) 

            # Causal Merge: Merge the 15-min feature onto the 3-min DataFrame
            df = df.merge(df_15m_features, left_index=True, right_index=True, how='left')
        else:
            df['ema40_slope'] = 0.0

        if 'ema40_slope' not in df.columns: df['ema40_slope'] = 0.0        
        df['macro_trend_slope'] = df['ema40_slope'] / df['atr']
        
        # --- 4. Final Cleanup ---
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(method='ffill', inplace=True); df.fillna(method='bfill', inplace=True); df.fillna(0, inplace=True)

        all_feature_cols = self.get_feature_columns()
        for col in all_feature_cols:
            if col not in df.columns: 
                df[col] = 0.0
                    
        logging.debug(f"V3.10 Supertrend features added. Shape after features: {df.shape}")
        return df

    # =========================================================
    # LIVE EXECUTION FUNCTIONS
    # (The live predict/load/enter logic remains unchanged as it is clean)
    # =========================================================

    def load_model(self):
        """Load ONNX model for Supertrend Pullback (V3.10)."""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            self.model = onnxruntime.InferenceSession(self.model_path)
            logging.info(f"✅ Loaded Supertrend Pullback V3.10 model: {os.path.basename(self.model_path)}")
        except Exception as e:
            logging.exception(f"❌ Error loading Supertrend Pullback V3.10 model: {e}")
            raise

    def load_scaler(self):
        """Load scaler for Supertrend Pullback (V3.10)."""
        try:
            if not os.path.exists(self.scaler_path):
                raise FileNotFoundError(f"Scaler file not found: {self.scaler_path}")

            with open(self.scaler_path, 'rb') as f:
                scalers = pickle.load(f)

            base_symbol = self.contract_symbol.split('.')[0][:2] 
            
            if base_symbol in scalers:
                self.scaler = scalers[base_symbol]
                logging.info(f"✅ Loaded '{base_symbol}' scaler for Supertrend V3.10")
            else:
                available = list(scalers.keys())
                raise ValueError(
                    f"'{base_symbol}' scaler not found. "
                    f"Available: {available}"
                )
        except Exception as e:
            logging.exception(f"❌ Error loading Supertrend V3.10 scaler: {e}")
            raise

    # --- (No change to predict() function, it's correct) ---
    def predict(self, df: pd.DataFrame) -> Tuple[int, float]:
        """
        Generate prediction using V3.10 Transformer model.
        """
        try:
            seq_len = self.get_sequence_length() # 120
            if df.empty or len(df) < seq_len:
                logging.warning(f"⚠️ Not enough data for prediction. Need {seq_len}, have {len(df)}. (Warm-up in progress)")
                return 0, 0.0

            features = self.preprocess_features(df) 

            if len(features) < seq_len:
                logging.warning(f"⚠️ Data length mismatch after scaling. Need {seq_len}, have {len(features)}. Returning Hold.")
                return 0, 0.0

            X = features[-seq_len:].reshape(1, seq_len, -1).astype(np.float32)

            input_name = self.model.get_inputs()[0].name
            output_name = self.model.get_outputs()[0].name
            logits_sequence = self.model.run([output_name], {input_name: X})[0] 

            last_logits = logits_sequence[0, -1, :] 

            probs = self._softmax(last_logits)
            prediction = int(np.argmax(probs))
            confidence = float(probs[prediction])

            return prediction, confidence

        except Exception as e:
            logging.exception(f"❌ Prediction error (Supertrend V3.10): {e}")
            return 0, 0.0 # Return Hold on error
    
        
    def should_enter_trade(
        self,
        prediction: int,
        confidence: float,
        bar: Dict, 
        entry_conf: float,
        adx_thresh: float, # Argument is ignored in Pure AI Mode       
    ) -> Tuple[bool, Optional[str]]:
        """
        Determine if V3.10 "Pure AI Mode" entry conditions are met.
        """
        
        # 1. Confidence Filter (Primary Filter)
        if confidence < entry_conf:
            return False, None
        
        # 2. Get fundamental regime state
        st_direction = bar.get('st_direction', 0.0)
            
        # 3. Model Prediction + Regime Filter
        if prediction == 1: # Model wants to BUY
            # Check 1: Is the Supertrend regime currently LONG?
            if st_direction == 1:
                return True, 'LONG'
                
        elif prediction == 2: # Model wants to SELL
            # Check 1: Is the Supertrend regime currently SHORT?
            if st_direction == -1:
                return True, 'SHORT'

        # Prediction is 0 (Hold) or alignment filter failed
        return False, None

    @staticmethod
    def _softmax(x):
        """Compute softmax values for array x."""
        if x is None or len(x) == 0:
            return np.array([])
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
