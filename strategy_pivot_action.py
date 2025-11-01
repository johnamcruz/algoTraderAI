#!/usr/bin/env python3
"""
Pivot Action Strategy Implementation (V2.8 - Risk Aware)

This strategy uses a Transformer model to trade pivot-based signals
(both reversals and pullbacks) based on the V2.8 "Pivot Action"
training, which uses a 2:1 R:R, 19 features, and an EMA50 proximity filter.
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


# CLASS NAME UPDATED TO 'PivotAction3minStrategy'
class PivotAction3minStrategy(BaseStrategy):
    """
    Pivot Action V2.8 (Risk-Aware Transformer) trading strategy.
    Implements 120-bar context, 19 features, and an EMA50 proximity filter.
    """

    def __init__(self, model_path: str, scaler_path: str, contract_symbol: str, 
                 pivot_lookback: int = 10, 
                 proximity_thresh: float = 0.35): # Proximity threshold from V2.8
        """
        Initialize Pivot Action V2.8 strategy.
        """
        super().__init__(model_path, scaler_path, contract_symbol)
        self.pivot_lookback = pivot_lookback
        self.proximity_thresh = proximity_thresh # Store proximity threshold
        logging.info(f"Initialized PivotAction3minStrategy V2.8 (Risk-Aware)")
        logging.info(f"V2.8 Proximity Threshold set to: {self.proximity_thresh}")


    def get_feature_columns(self) -> List[str]:
        """
        Returns the 19 feature columns for the V2.8 Transformer.
        (Identical to V2.5)
        """
        return [
            'dist_to_pivot', 'bars_since_pivot', 'pivot_type',
            'price_vel_10', 'price_vel_20', 'rsi', 'rsi_vel_10',
            'body_size', 'wick_ratio', 'rejection_wick',
            'price_vs_ema50','price_vs_ema200', 'ema50_vs_ema200', 'adx',
            'change_since_confirm', 
            'macro_trend_slope',
            # V2.5/V2.8 NEW FEATURES
            'price_roc_slope',          
            'inverse_volatility_score', 
            'volume_velocity'           
        ]

    def get_sequence_length(self) -> int:
        """Pivot Action V2.8 (Transformer) uses 120 bars."""
        return 120

    # =========================================================
    # V2.8 FEATURE ENGINEERING (Identical to V2.5)
    # =========================================================
    
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Pivot Action V2.8 features (19 total).
        """        
        logging.debug(f"Adding V2.8 features. Input shape: {df.shape}")
        
        df = df.copy()

        # === FIX: CLEAN OHLCV DATA FIRST ===
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(method='ffill')
            df[col] = df[col].fillna(0)             

        n = self.pivot_lookback

        # === CORE INDICATORS ===
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['atr'] = df['atr'].replace(0, 1e-6)
        df['atr'] = df['atr'].fillna(method='ffill')
        df['atr'] = df['atr'].fillna(1e-6) 

        df['ema9'] = ta.ema(df['close'], length=9)
        df['ema21'] = ta.ema(df['close'], length=21)
        df['ema50'] = ta.ema(df['close'], length=50)
        df['ema200'] = ta.ema(df['close'], length=200)

        adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
        df['adx'] = adx_df['ADX_14'] if adx_df is not None and 'ADX_14' in adx_df.columns else 0
        
        df['rsi'] = ta.rsi(df['close'], length=14)
        
        # === PIVOT IDENTIFICATION (V2.0 LOGIC) ===
        df['pivot_val'] = self._find_pivots(df['close'], n, n) 
        df['pivot_type'] = np.sign(df['pivot_val']).fillna(method='ffill').fillna(0)
        df['pivot_val'] = df['pivot_val'].abs().fillna(method='ffill').fillna(0)

        df['bars_since_pivot'] = df.groupby(
            (df['pivot_val'] != df['pivot_val'].shift(1)).cumsum()
        ).cumcount()
        
        df['dist_to_pivot'] = (df['pivot_val'] - df['close']) / df['atr'] * df['pivot_type'] 
        
        # OPTIMIZATION 1: CHANGE SINCE CONFIRMATION (CSC)
        df['change_since_confirm'] = (df['close'] - df['close'].shift(n)) / df['atr']
        
        # === V2.0: RELATIVE VELOCITY & DIVERGENCE FEATURES ===
        df['price_vel_10'] = df['close'].diff(10) / df['atr']
        df['price_vel_20'] = df['close'].diff(20) / df['atr']
        df['rsi_vel_10'] = df['rsi'].diff(10) 

        # === V2.0: STRUCTURAL CONTEXT FEATURES ===
        df['price_vs_ema50'] = (df['close'] - df['ema50']) / df['atr']
        df['price_vs_ema200'] = (df['close'] - df['ema200']) / df['atr']
        df['ema50_vs_ema200'] = (df['ema50'] - df['ema200']) / df['atr']

        # === V2.5/V2.8 ENHANCED RISK/MOMENTUM FEATURES (3 NEW) ===
    
        # 1. Price Rate of Change (ROC) Slope / Momentum Decay
        df['price_roc_slope'] = df['price_vel_10'].diff(10) 
        
        # 2. Inverse Volatility Score (IVS) - Proxy for Compression
        df['vol_channel_width'] = (df['ema9'] - df['ema21']).abs() 
        df['inverse_volatility_score'] = df['vol_channel_width'] / (df['atr'] + 1e-6)
    
        # 3. Volume Velocity (Confirmation)
        vol_std = df['volume'].rolling(10).std().replace(0, 1e-6)
        df['volume_velocity'] = df['volume'].diff(1) / vol_std

        # === V2.0: CANDLESTICK FEATURES ===
        df['body_size'] = abs(df['close'] - df['open']) / df['atr']
        df['wick_ratio'] = (df['high'] - df['low']) / df['atr'] 
        df['rejection_wick'] = np.where(df['close'] > df['open'], 
                                    (df['close'] - df['low']) / df['atr'], 
                                    (df['high'] - df['close']) / df['atr']) 

        # OPTIMIZATION 2: 15-MIN MACRO CONTEXT
        df_15m = df[['close']].resample('15Min', label='left', closed='left').ohlc().dropna()
        df_15m.columns = df_15m.columns.droplevel(0) 
        
        df_15m['ema40'] = ta.ema(df_15m['close'], length=40)
        df_15m['ema40'] = df_15m['ema40'].fillna(0) # FIX
        
        df_15m['ema40_slope'] = df_15m['ema40'].diff(3) 
        df_15m = df_15m[['ema40_slope']]
        
        df = df.merge(df_15m, left_index=True, right_index=True, how='left')
        df['ema40_slope'] = df['ema40_slope'].fillna(method='ffill')
        df['ema40_slope'] = df['ema40_slope'].fillna(0) 
        
        df['macro_trend_slope'] = df['ema40_slope'] / df['atr']

        # === FINAL CLEANUP ===
        all_feature_cols = self.get_feature_columns()
        for col in all_feature_cols:
            if col not in df.columns:
                df[col] = 0.0
                
        df[all_feature_cols] = df[all_feature_cols].replace([np.inf, -np.inf], np.nan)
        df[all_feature_cols] = df[all_feature_cols].fillna(method='ffill') 
        df[all_feature_cols] = df[all_feature_cols].fillna(0) 

        logging.debug(f"V2.8 features added. Shape after features: {df.shape}")
        return df

    def _find_pivots(self, series: pd.Series, n_left: int, n_right: int) -> pd.Series:
        """
        V2.0 find_pivots logic WITH NON-REPAINTING FIX
        """
        pivots = pd.Series(np.nan, index=series.index)
        
        if len(series) < n_left + n_right + 1:
            logging.warning("Not enough data to find pivots.")
            return pivots

        for i in range(n_left, len(series) - n_right):
            is_pivot_h = all(series.iloc[i] > series.iloc[i - k] for k in range(1, n_left + 1)) and \
                         all(series.iloc[i] >= series.iloc[i + k] for k in range(1, n_right + 1))
            is_pivot_l = all(series.iloc[i] < series.iloc[i - k] for k in range(1, n_left + 1)) and \
                         all(series.iloc[i] <= series.iloc[i + k] for k in range(1, n_right + 1))

            if is_pivot_h:
                pivots.iloc[i] = series.iloc[i]
            elif is_pivot_l:
                pivots.iloc[i] = series.iloc[i] * -1
                
        pivots = pivots.shift(n_right)
        return pivots

    # =========================================================
    # LIVE EXECUTION FUNCTIONS
    # =========================================================

    def load_model(self):
        """Load the V2.8 ONNX model."""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            self.model = onnxruntime.InferenceSession(self.model_path)
            logging.info(f"✅ Loaded Pivot Action V2.8 model: {os.path.basename(self.model_path)}")
        except Exception as e:
            logging.exception(f"❌ Error loading Pivot Action V2.8 model: {e}")
            raise

    def load_scaler(self):
        """Load the V2.8 scaler.pkl file."""
        try:
            if not os.path.exists(self.scaler_path):
                raise FileNotFoundError(f"Scaler file not found: {self.scaler_path}")
            with open(self.scaler_path, 'rb') as f:
                scalers = pickle.load(f)
            base_symbol = self.contract_symbol.split('.')[0][:2] 
            if base_symbol in scalers:
                self.scaler = scalers[base_symbol]
                logging.info(f"✅ Loaded '{base_symbol}' scaler for Pivot Action V2.8")
            else:
                available = list(scalers.keys())
                raise ValueError(f"'{base_symbol}' scaler not found. Available: {available}")
        except Exception as e:
            logging.exception(f"❌ Error loading Pivot Action V2.8 scaler: {e}")
            raise

    def predict(self, df: pd.DataFrame) -> Tuple[int, float]:
        """
        Generate prediction using V2.8 Transformer model.
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
            last_logits = logits_sequence[0, -1, :] # Shape: (3,)

            probs = self._softmax(last_logits)
            prediction = int(np.argmax(probs))
            confidence = float(probs[prediction])

            return prediction, confidence

        except Exception as e:
            logging.exception(f"❌ Prediction error (Pivot Action V2.8): {e}")
            return 0, 0.0 

    def should_enter_trade(
        self,
        prediction: int,
        confidence: float,
        bar: Dict, # Bar must contain features for filtering
        entry_conf: float,
        adx_thresh: float # This parameter is ignored in V2.8
    ) -> Tuple[bool, Optional[str]]:
        """
        V2.8 UPDATE: Determine if entry conditions are met.
        Uses the EMA50 for proximity filtering to match V2.8 training.
        """
        
        # 1. V2.8 PROXIMITY FILTER (THE CRITICAL FIX)
        # The V2.8 model was trained on signals near the EMA50.
        curr_close = bar.get('close')
        pivot_line = bar.get('ema50') # <-- V2.8 uses ema50
        atr = bar.get('atr')

        # Safety check for missing data
        if curr_close is None or pivot_line is None or atr is None or atr < 1e-6:
            logging.warning("Missing data for V2.8 proximity check (close, ema50, or atr).")
            return False, None
            
        # Calculate proximity to the EMA50
        dist_to_pivot_line = abs(curr_close - pivot_line) / atr
        
        if dist_to_pivot_line > self.proximity_thresh:
            # logging.debug(f"Trade filtered by proximity: {dist_to_pivot_line:.2f} > {self.proximity_thresh}")
            return False, None # Not close enough to the pivot line (EMA50)

        # 2. Confidence Filter
        if confidence < entry_conf:
            # logging.debug(f"Trade filtered by confidence: {confidence:.2f} < {entry_conf}")
            return False, None

        # 3. Model Prediction Filter
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