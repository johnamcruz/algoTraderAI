#!/usr/bin/env python3
"""
Supertrend Pullback Strategy Implementation (V3.0 - Transformer)

This strategy uses a Transformer model to trade Supertrend pullbacks.
It is based on the V3.0 "Classification-Only, Filtered Labeling" model.
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
    Supertrend Pullback V3.0 (Transformer) trading strategy.
    Uses ADX and EMA200 filters for high-precision entries.
    """

    def __init__(self, model_path: str, scaler_path: str, contract_symbol: str):
        """
        Initialize Supertrend Pullback V3.0 strategy.
        """
        # pivot_lookback is no longer needed for this strategy
        super().__init__(model_path, scaler_path, contract_symbol)
        logging.info("Initialized SupertrendPullbackStrategy")


    def get_feature_columns(self) -> List[str]:
        """
        UPDATED: Returns the 15 feature columns for the V3 Supertrend Transformer.
        """
        return [ 
            'price_vs_st', 'st_direction', 'price_vs_ema40', 'ema15_vs_ema40', 'price_vs_ema200',
            'adx', 'adx_slope', 'rsi', 'cmf',
            'price_vel_10', 'price_vel_20', 'rsi_vel_10', 
            'body_size', 'wick_ratio', 'atr'
        ]

    def get_sequence_length(self) -> int:
        """Supertrend Pullback V3.0 (Transformer) uses 80 bars."""
        return 80

    # =========================================================
    # V3.0 FEATURE ENGINEERING (REPLACED)
    # =========================================================
    
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        UPDATED: Calculate Supertrend Pullback V3.0 features.
        """        
        logging.debug(f"Adding V3 Supertrend features. Input shape: {df.shape}")
        
        df = df.copy()

        # === CLEAN OHLCV DATA FIRST ===
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(method='ffill')
            df[col] = df[col].fillna(0)             

        # === CORE INDICATORS & VOLATILITY ===
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['atr'] = df['atr'].replace(0, 1e-6)
        df['atr'] = df['atr'].fillna(method='ffill')
        df['atr'] = df['atr'].fillna(1e-6) 

        df['ema15'] = ta.ema(df['close'], length=15)
        df['ema40'] = ta.ema(df['close'], length=40)
        df['ema200'] = ta.ema(df['close'], length=200)

        # === SUPERTRND (ST) IDENTIFICATION ===
        st_df = ta.supertrend(df['high'], df['low'], df['close'], length=10, multiplier=3)
        df['st_val'] = st_df['SUPERT_10_3']
        df['st_direction'] = st_df['SUPERTd_10_3'] 
        
        # === RELATIVE POSITION & CONTEXT ===
        df['price_vs_st'] = (df['close'] - df['st_val']) / df['atr']
        df['price_vs_ema40'] = (df['close'] - df['ema40']) / df['atr']
        df['ema15_vs_ema40'] = (df['ema15'] - df['ema40']) / df['atr']
        df['price_vs_ema200'] = (df['close'] - df['ema200']) / df['atr'] # Feature for macro filter
        
        # === MOMENTUM & VOLUME ===
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
        df['adx'] = adx_df['ADX_14'] if adx_df is not None and 'ADX_14' in adx_df.columns else 0
        df['adx_slope'] = df['adx'].diff(5) 
        df['rsi'] = ta.rsi(df['close'], length=14)
        df['cmf'] = ta.cmf(df['high'], df['low'], df['close'], df['volume'], length=20)
        
        # === VELOCITY FEATURES (from Pivot V2.0 Success) ===
        df['price_vel_10'] = df['close'].diff(10) / df['atr']
        df['price_vel_20'] = df['close'].diff(20) / df['atr']
        df['rsi_vel_10'] = df['rsi'].diff(10) 
        
        # === CANDLESTICK FEATURES ===
        df['body_size'] = abs(df['close'] - df['open']) / df['atr']
        df['wick_ratio'] = (df['high'] - df['low']) / df['atr'] 

        # === FINAL CLEANUP ===
        all_feature_cols = self.get_feature_columns()
        for col in all_feature_cols:
            if col not in df.columns:
                df[col] = 0.0
                
        df[all_feature_cols] = df[all_feature_cols].replace([np.inf, -np.inf], np.nan)
        df[all_feature_cols] = df[all_feature_cols].fillna(method='ffill') # Fill gaps
        df[all_feature_cols] = df[all_feature_cols].fillna(0) # Fill leading NaNs

        logging.debug(f"V3 Supertrend features added. Shape after features: {df.shape}")
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
            logging.info(f"✅ Loaded Supertrend Pullback V3 model: {os.path.basename(self.model_path)}")
        except Exception as e:
            logging.exception(f"❌ Error loading Supertrend Pullback V3 model: {e}")
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
                logging.info(f"✅ Loaded '{base_symbol}' scaler for Supertrend V3")
            else:
                available = list(scalers.keys())
                raise ValueError(
                    f"'{base_symbol}' scaler not found. "
                    f"Available: {available}"
                )
        except Exception as e:
            logging.exception(f"❌ Error loading Supertrend V3 scaler: {e}")
            raise

    def predict(self, df: pd.DataFrame) -> Tuple[int, float]:
        """
        Generate prediction using V3 Transformer model (Classification-Only).
        """
        try:
            # preprocess_features is assumed to be in BaseStrategy
            features = self.preprocess_features(df) 

            seq_len = self.get_sequence_length() # 80
            if len(features) < seq_len:
                logging.warning(f"⚠️ Not enough data for prediction. Need {seq_len}, have {len(features)}. Returning Hold.")
                return 0, 0.0

            X = features[-seq_len:].reshape(1, seq_len, -1).astype(np.float32)

            input_name = self.model.get_inputs()[0].name
            output_name = self.model.get_outputs()[0].name
            logits_sequence = self.model.run([output_name], {input_name: X})[0] # Shape: (1, 80, 3)

            # Get logits from the LAST time step
            last_logits = logits_sequence[0, -1, :] # Shape: (3,)

            probs = self._softmax(last_logits)
            prediction = int(np.argmax(probs))
            confidence = float(probs[prediction])

            return prediction, confidence

        except Exception as e:
            logging.exception(f"❌ Prediction error (Supertrend V3): {e}")
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
        Determine if Supertrend V3 entry conditions are met.
        Applies hard filters for ADX (chop) and EMA (macro-trend).
        """
        
        # 1. Confidence Filter
        if confidence < entry_conf:
            return False, None
        
        # Use .get() and ensure the value is not None before proceeding with comparison
        # Using 0.0 as a safe numeric fallback for all keys in comparison

        # Get values with safe fallback if key is missing or value is None
        close = bar.get('close') if bar.get('close') is not None else 0.0
        ema200 = bar.get('ema200') if bar.get('ema200') is not None else 0.0
        st_direction = bar.get('st_direction') if bar.get('st_direction') is not None else 0.0
        adx = bar.get('adx') if bar.get('adx') is not None else 0.0
        
        # 2. Chop Filter (Uses the passed adx_thresh)
        if adx_thresh > 0 and adx < adx_thresh:
            return False, None # Market is choppy
            
        # 3. Model Prediction Filter
        if prediction == 1: # Model wants to BUY
            # 4. Macro-Trend & Alignment Filters
            # Check 1: close > ema200 AND Check 2: st_direction == 1
            if close > ema200 and st_direction == 1: # No more comparison with NoneType
                return True, 'LONG'
                
        elif prediction == 2: # Model wants to SELL
            # 4. Macro-Trend & Alignment Filters
            # Check 1: close < ema200 AND Check 2: st_direction == -1
            if close < ema200 and st_direction == -1: # No more comparison with NoneType
                return True, 'SHORT'

        # Prediction is 0 (Hold) or filters failed
        return False, None

    @staticmethod
    def _softmax(x):
        """Compute softmax values for array x."""
        if x is None or len(x) == 0:
            return np.array([])
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()