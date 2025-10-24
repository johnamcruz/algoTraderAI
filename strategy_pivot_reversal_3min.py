#!/usr/bin/env python3
"""
Pivot Reversal Strategy Implementation (3-Minute Timeframe Version)

This strategy uses pivot point breaks and rejections with momentum analysis,
adapted for a 3-minute timeframe based on equivalent 5-minute durations.
"""

import pandas as pd
import pandas_ta as ta
import numpy as np
import onnxruntime
import pickle
import os
from typing import Dict, List, Tuple, Optional
# Assuming strategy_base.py contains the BaseStrategy class
from strategy_base import BaseStrategy


class PivotReversal3minStrategy(BaseStrategy):
    """
    Pivot Reversal trading strategy based on pivot breaks/rejections.
    (Optimized for 3-Minute Timeframe Parameters)
    """

    # UPDATED: Default pivot_lookback changed from 5 to 8
    def __init__(self, model_path: str, scaler_path: str, contract_symbol: str, pivot_lookback: int = 8):
        """
        Initialize Pivot Reversal strategy for 3-minute timeframe.

        Args:
            model_path: Path to ONNX model (trained on 3-min data with seq_len=67)
            scaler_path: Path to scaler (saved from 3-min training)
            contract_symbol: Trading symbol
            pivot_lookback: Number of bars for pivot detection (default 8 for 3-min)
        """
        super().__init__(model_path, scaler_path, contract_symbol)
        # Store the pivot lookback specific to this instance
        self.pivot_lookback = pivot_lookback
        print(f"** Initializing PivotReversalStrategy for {contract_symbol} (3-min TF) with pivot_lookback={self.pivot_lookback} **")


    def get_feature_columns(self) -> List[str]:
        """Returns feature columns for Pivot Reversal strategy."""
        # Feature columns remain the same
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

    # UPDATED: Sequence length changed from 40 to 67
    def get_sequence_length(self) -> int:
        """Pivot Reversal uses 67 bars for the 3-minute timeframe model."""
        return 67

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Pivot Reversal features - FIXED NON-REPAINTING LOGIC.
        Uses the instance's self.pivot_lookback.
        """
        print(f"  Adding features with pivot_lookback={self.pivot_lookback}...") # Added print for clarity
        # === FIX: CLEAN OHLCV DATA FIRST (NO INPLACE=TRUE) ===
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(method='ffill') # Fill gaps
            df[col] = df[col].fillna(0)             # Fill leading NaNs

        # Use the lookback set during initialization
        n = self.pivot_lookback

        # === CORE INDICATORS (NO INPLACE=TRUE) ===
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['atr'] = df['atr'].replace(0, 1e-6)
        df['atr'] = df['atr'].fillna(method='ffill')
        df['atr'] = df['atr'].fillna(1e-6)

        df['ema9'] = ta.ema(df['close'], length=9)
        df['ema21'] = ta.ema(df['close'], length=21)
        df['ema50'] = ta.ema(df['close'], length=50)

        df['ema9'] = df['ema9'].fillna(method='ffill'); df['ema9'] = df['ema9'].fillna(0)
        df['ema21'] = df['ema21'].fillna(method='ffill'); df['ema21'] = df['ema21'].fillna(0)
        df['ema50'] = df['ema50'].fillna(method='ffill'); df['ema50'] = df['ema50'].fillna(0)

        adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
        df['adx'] = adx_df['ADX_14'] if adx_df is not None and 'ADX_14' in adx_df.columns else 0
        df['adx'] = df['adx'].fillna(method='ffill'); df['adx'] = df['adx'].fillna(0)

        df['rsi'] = ta.rsi(df['close'], length=14)
        df['rsi'] = df['rsi'].fillna(method='ffill'); df['rsi'] = df['rsi'].fillna(50)

        stoch_df = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3, smooth_k=3)
        df['stoch_k'] = stoch_df['STOCHk_14_3_3'] if stoch_df is not None and 'STOCHk_14_3_3' in stoch_df.columns else 50
        df['stoch_k'] = df['stoch_k'].fillna(method='ffill'); df['stoch_k'] = df['stoch_k'].fillna(50)

        macd_df = ta.macd(df['close'], fast=12, slow=26, signal=9)
        df['macd_hist'] = macd_df['MACDh_12_26_9'] if macd_df is not None and 'MACDh_12_26_9' in macd_df.columns else 0
        df['macd_hist'] = df['macd_hist'].fillna(method='ffill'); df['macd_hist'] = df['macd_hist'].fillna(0)

        # === PIVOT IDENTIFICATION (FIXED: NON-REPAINTING & CORRECT LOGIC) ===
        # Uses self.pivot_lookback (n) passed during init
        df['pivot_high_val'] = self._find_pivots(df['high'], n)
        df['pivot_low_val'] = self._find_pivots(df['low'] * -1, n) * -1

        df['pivot_high_val'] = df['pivot_high_val'].fillna(method='ffill')
        df['pivot_low_val'] = df['pivot_low_val'].fillna(method='ffill')
        df['pivot_high_val'] = df['pivot_high_val'].fillna(df['high'])
        df['pivot_low_val'] = df['pivot_low_val'].fillna(df['low'])

        df['bars_since_ph'] = df.groupby(
            (df['pivot_high_val'].notna() &
             (df['pivot_high_val'] != df['pivot_high_val'].shift(1))).cumsum()
        ).cumcount()
        df['bars_since_pl'] = df.groupby(
            (df['pivot_low_val'].notna() &
             (df['pivot_low_val'] != df['pivot_low_val'].shift(1))).cumsum()
        ).cumcount()

        df['dist_to_ph'] = (df['pivot_high_val'] - df['close']) / df['atr']
        df['dist_to_pl'] = (df['close'] - df['pivot_low_val']) / df['atr']

        # === PIVOT BREAK / REVERSAL SIGNALS ===
        broke_ph_cond = (
            (df['close'] > df['pivot_high_val'].shift(1)) &
            (df['close'] - df['pivot_high_val'].shift(1) > 0.25 * df['atr'])
        )
        broke_pl_cond = (
            (df['close'] < df['pivot_low_val'].shift(1)) &
            (df['pivot_low_val'].shift(1) - df['close'] > 0.25 * df['atr'])
        )

        # === CANDLE CHARACTERISTICS (NO INPLACE=TRUE) ===
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
        df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']
        df['total_range'] = df['high'] - df['low']
        df['total_range'] = df['total_range'].replace(0, 1e-6)

        df['upper_wick_ratio'] = df['upper_wick'] / df['total_range']
        df['lower_wick_ratio'] = df['lower_wick'] / df['total_range']
        df['body_ratio'] = df['body_size'] / df['total_range']

        is_bullish_rejection_candle = (
            (df['lower_wick_ratio'] > 0.5) & (df['body_ratio'] < 0.3)
        )
        is_bearish_rejection_candle = (
            (df['upper_wick_ratio'] > 0.5) & (df['body_ratio'] < 0.3)
        )

        near_ph = abs(df['high'] - df['pivot_high_val']) < 0.25 * df['atr']
        near_pl = abs(df['low'] - df['pivot_low_val']) < 0.25 * df['atr']
        reject_at_ph_cond = is_bearish_rejection_candle & near_ph
        reject_at_pl_cond = is_bullish_rejection_candle & near_pl

        # === CONTEXT FEATURES ===
        df['price_vs_ema50'] = (df['close'] - df['ema50']) / df['atr']
        is_uptrend_cond = (df['close'] > df['ema50']) & (df['ema9'] > df['ema21'])
        is_downtrend_cond = (df['close'] < df['ema50']) & (df['ema9'] < df['ema21'])
        is_trending_cond = (df['adx'] > 20)
        df['rsi_slope'] = df['rsi'].diff(3)
        df['stoch_k_slope'] = df['stoch_k'].diff(3)
        df['macd_hist_slope'] = df['macd_hist'].diff(3)

        # Convert boolean conditions to float for model
        df['broke_ph'] = broke_ph_cond.astype(float)
        df['broke_pl'] = broke_pl_cond.astype(float)
        df['reject_at_ph'] = reject_at_ph_cond.astype(float)
        df['reject_at_pl'] = reject_at_pl_cond.astype(float)
        df['bullish_rejection_candle'] = is_bullish_rejection_candle.astype(float)
        df['bearish_rejection_candle'] = is_bearish_rejection_candle.astype(float)
        df['is_uptrend'] = is_uptrend_cond.astype(float)
        df['is_downtrend'] = is_downtrend_cond.astype(float)
        df['is_trending'] = is_trending_cond.astype(float)

        # Clean up (FIXED: NO INPLACE=TRUE, applied to whole df)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill') # Fill gaps
        df = df.fillna(0) # Fill leading NaNs

        return df

    def _find_pivots(self, series: pd.Series, n: int) -> pd.Series:
        """Find pivot points (highs or lows) - SLOW, BUT NON-REPAINTING."""
        pivots = pd.Series(np.nan, index=series.index)
        # Ensure loop bounds are valid
        if len(series) < 2 * n + 1:
            return pivots # Not enough data to find pivots with this lookback

        for i in range(n, len(series) - n):
            is_pivot = True
            # Check left
            for k in range(1, n + 1):
                # Ensure index i-k is valid
                if i - k < 0:
                    is_pivot = False; break
                if series.iloc[i] <= series.iloc[i - k]:
                    is_pivot = False; break
            if not is_pivot: continue

            # Check right
            for k in range(1, n + 1):
                 # Ensure index i+k is valid
                if i + k >= len(series):
                    is_pivot = False; break
                if series.iloc[i] < series.iloc[i + k]:
                    is_pivot = False; break
            if is_pivot:
                pivots.iloc[i] = series.iloc[i]

        return pivots.shift(n) # Apply shift for non-repainting

    def load_model(self):
        """Load ONNX model for Pivot Reversal."""
        # --- (No changes needed here) ---
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
        # --- (No changes needed here) ---
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
        # --- (No changes needed here, uses self.get_sequence_length()) ---
        try:
            # Preprocess features
            features = self.preprocess_features(df) # Assumes preprocess_features exists in BaseStrategy

            # Prepare input for ONNX
            seq_len = self.get_sequence_length() # Will get 67 from the updated method
            if len(features) < seq_len:
                # Instead of raising error, maybe return neutral prediction?
                # For live trading, might be better to wait for enough data.
                print(f"⚠️ Not enough data for prediction. Need {seq_len}, have {len(features)}. Returning Hold.")
                return 0, 0.0 # Return Hold (0) with 0 confidence
                # raise ValueError(f"Not enough data. Need {seq_len}, have {len(features)}")

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
            # Optionally log the full traceback here
            # import traceback
            # print(traceback.format_exc())
            return 0, 0.0 # Return Hold on error

    def should_enter_trade(
        self,
        prediction: int,
        confidence: float,
        bar: Dict, # Assuming bar contains the latest data
        entry_conf: float,
        adx_thresh: float # Removed as per previous decision
    ) -> Tuple[bool, Optional[str]]:
        """Determine if Pivot Reversal entry conditions are met (No ADX filter)."""
        # --- (Removed adx_thresh parameter and check) ---

        # Check confidence threshold
        if confidence < entry_conf:
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
        # --- (No changes needed here) ---
        # Added check for empty input for safety
        if x is None or len(x) == 0:
            return np.array([])
        exp_x = np.exp(x - np.max(x)) # Subtract max for numerical stability
        return exp_x / exp_x.sum()