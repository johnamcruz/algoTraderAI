#!/usr/bin/env python3
"""
Pivot Reversal Strategy Implementation

This strategy uses pivot point breaks and rejections with momentum analysis.
"""

import pandas as pd
import pandas_ta as ta
import numpy as np
import onnxruntime
import pickle
import os
from typing import Dict, List, Tuple, Optional
from strategy_base import BaseStrategy


class PivotReversalStrategy(BaseStrategy):
    """
    Pivot Reversal trading strategy based on pivot breaks/rejections.
    """
    
    def __init__(self, model_path: str, scaler_path: str, contract_symbol: str, pivot_lookback: int = 5):
        """
        Initialize Pivot Reversal strategy.
        
        Args:
            model_path: Path to ONNX model
            scaler_path: Path to scaler
            contract_symbol: Trading symbol
            pivot_lookback: Number of bars for pivot detection
        """
        super().__init__(model_path, scaler_path, contract_symbol)
        self.pivot_lookback = pivot_lookback
    
    def get_feature_columns(self) -> List[str]:
        """Returns feature columns for Pivot Reversal strategy."""
        return [
            # Pivot Features
            'bars_since_ph', 'bars_since_pl', 'dist_to_ph', 'dist_to_pl',
            # Pivot Break/Rejection Signals
            'broke_ph', 'broke_pl', 'reject_at_ph', 'reject_at_pl',
            'bullish_rejection_candle', 'bearish_rejection_candle',
            'upper_wick_ratio', 'lower_wick_ratio', 'body_ratio',
            # Trend Context
            'price_vs_ema50', 'is_uptrend', 'is_downtrend', 'is_trending', 'adx',
            # Momentum Context
            'rsi', 'rsi_slope', 'stoch_k', 'stoch_k_slope', 'macd_hist', 'macd_hist_slope'
        ]
    
    def get_sequence_length(self) -> int:
        """Pivot Reversal uses 40 bars."""
        return 40
    
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Pivot Reversal features."""
        # Ensure numeric types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        # === CORE INDICATORS ===
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['atr'].replace(0, 1e-6, inplace=True)
        df['ema9'] = ta.ema(df['close'], length=9)
        df['ema21'] = ta.ema(df['close'], length=21)
        df['ema50'] = ta.ema(df['close'], length=50)
        
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
        df['adx'] = adx_df['ADX_14']
        df['rsi'] = ta.rsi(df['close'], length=14)
        
        stoch_df = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3, smooth_k=3)
        df['stoch_k'] = stoch_df['STOCHk_14_3_3']
        
        macd_df = ta.macd(df['close'], fast=12, slow=26, signal=9)
        df['macd_hist'] = macd_df['MACDh_12_26_9']

        # === PIVOT IDENTIFICATION ===
        df['pivot_high_val'] = self._find_pivots(df['high'], self.pivot_lookback)
        df['pivot_low_val'] = self._find_pivots(df['low'], self.pivot_lookback)
        df['pivot_high_val'].fillna(method='ffill', inplace=True)
        df['pivot_low_val'].fillna(method='ffill', inplace=True)
        
        # Bars since pivot
        df['bars_since_ph'] = df.groupby(
            (df['pivot_high_val'].notna() & 
             (df['pivot_high_val'] != df['pivot_high_val'].shift(1))).cumsum()
        ).cumcount()
        df['bars_since_pl'] = df.groupby(
            (df['pivot_low_val'].notna() & 
             (df['pivot_low_val'] != df['pivot_low_val'].shift(1))).cumsum()
        ).cumcount()
        
        # Distance to pivots
        df['dist_to_ph'] = (df['pivot_high_val'] - df['close']) / df['atr']
        df['dist_to_pl'] = (df['close'] - df['pivot_low_val']) / df['atr']

        # === PIVOT BREAK SIGNALS ===
        df['broke_ph'] = (
            (df['close'] > df['pivot_high_val'].shift(1)) & 
            (df['close'] - df['pivot_high_val'].shift(1) > 0.25 * df['atr'])
        ).astype(float)
        df['broke_pl'] = (
            (df['close'] < df['pivot_low_val'].shift(1)) & 
            (df['pivot_low_val'].shift(1) - df['close'] > 0.25 * df['atr'])
        ).astype(float)

        # === CANDLE CHARACTERISTICS ===
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
        df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']
        df['total_range'] = df['high'] - df['low']
        df['total_range'].replace(0, 1e-6, inplace=True)
        df['upper_wick_ratio'] = df['upper_wick'] / df['total_range']
        df['lower_wick_ratio'] = df['lower_wick'] / df['total_range']
        df['body_ratio'] = df['body_size'] / df['total_range']
        
        # Rejection candles
        is_bullish_rejection = (
            (df['lower_wick_ratio'] > 0.5) & (df['body_ratio'] < 0.3)
        )
        is_bearish_rejection = (
            (df['upper_wick_ratio'] > 0.5) & (df['body_ratio'] < 0.3)
        )
        df['bullish_rejection_candle'] = is_bullish_rejection.astype(float)
        df['bearish_rejection_candle'] = is_bearish_rejection.astype(float)
        
        # Rejection at pivots
        near_ph = abs(df['high'] - df['pivot_high_val']) < 0.25 * df['atr']
        near_pl = abs(df['low'] - df['pivot_low_val']) < 0.25 * df['atr']
        df['reject_at_ph'] = (is_bearish_rejection & near_ph).astype(float)
        df['reject_at_pl'] = (is_bullish_rejection & near_pl).astype(float)

        # === CONTEXT FEATURES ===
        df['price_vs_ema50'] = (df['close'] - df['ema50']) / df['atr']
        df['is_uptrend'] = (
            (df['close'] > df['ema50']) & (df['ema9'] > df['ema21'])
        ).astype(float)
        df['is_downtrend'] = (
            (df['close'] < df['ema50']) & (df['ema9'] < df['ema21'])
        ).astype(float)
        df['is_trending'] = (df['adx'] > 20).astype(float)
        df['rsi_slope'] = df['rsi'].diff(3)
        df['stoch_k_slope'] = df['stoch_k'].diff(3)
        df['macd_hist_slope'] = df['macd_hist'].diff(3)

        # Clean up
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        df.fillna(0, inplace=True)
        
        return df
    
    def _find_pivots(self, series: pd.Series, n: int) -> pd.Series:
        """Find pivot points (highs or lows)."""
        pivots = pd.Series(np.nan, index=series.index)
        for i in range(n, len(series) - n):
            is_pivot = True
            # Check left
            for k in range(1, n + 1):
                if series.iloc[i] <= series.iloc[i - k]:
                    is_pivot = False
                    break
            if not is_pivot:
                continue
            # Check right
            for k in range(1, n + 1):
                if series.iloc[i] < series.iloc[i + k]:
                    is_pivot = False
                    break
            if is_pivot:
                pivots.iloc[i] = series.iloc[i]
        return pivots
    
    def load_model(self):
        """Load ONNX model for Pivot Reversal."""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            self.model = onnxruntime.InferenceSession(self.model_path)
            print(f"✅ Loaded Pivot Reversal model: {os.path.basename(self.model_path)}")
        except Exception as e:
            print(f"❌ Error loading Pivot Reversal model: {e}")
            raise
    
    def load_scaler(self):
        """Load scaler for Pivot Reversal."""
        try:
            if not os.path.exists(self.scaler_path):
                raise FileNotFoundError(f"Scaler file not found: {self.scaler_path}")
            
            with open(self.scaler_path, 'rb') as f:
                scalers = pickle.load(f)
            
            if self.contract_symbol in scalers:
                self.scaler = scalers[self.contract_symbol]
                print(f"✅ Loaded '{self.contract_symbol}' scaler for Pivot Reversal")
            else:
                available = list(scalers.keys())
                raise ValueError(
                    f"'{self.contract_symbol}' scaler not found. "
                    f"Available: {available}"
                )
        except Exception as e:
            print(f"❌ Error loading Pivot Reversal scaler: {e}")
            raise
    
    def predict(self, df: pd.DataFrame) -> Tuple[int, float]:
        """Generate prediction using Pivot Reversal model."""
        try:
            # Preprocess features
            features = self.preprocess_features(df)
            
            # Prepare input for ONNX
            seq_len = self.get_sequence_length()
            if len(features) < seq_len:
                raise ValueError(f"Not enough data. Need {seq_len}, have {len(features)}")
            
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
            
            return prediction, confidence
            
        except Exception as e:
            print(f"❌ Prediction error (Pivot Reversal): {e}")
            return 0, 0.0
    
    def should_enter_trade(
        self, 
        prediction: int, 
        confidence: float,
        bar: Dict,
        entry_conf: float,
        adx_thresh: float
    ) -> Tuple[bool, Optional[str]]:
        """Determine if Pivot Reversal entry conditions are met."""
        
        # Check confidence threshold
        if confidence < entry_conf:
            return False, None
        
        # Check ADX threshold
        adx = bar.get('adx', 0)
        if adx_thresh > 0 and adx < adx_thresh:
            return False, None
        
        # Check prediction
        if prediction == 1:  # Buy signal
            return True, 'LONG'
        elif prediction == 2:  # Sell signal
            return True, 'SHORT'
        
        return False, None
    
    @staticmethod
    def _softmax(x):
        """Compute softmax values for array x."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()