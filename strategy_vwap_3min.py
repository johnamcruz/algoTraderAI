#!/usr/bin/env python3
"""
VWAP Mean Reversion Strategy Implementation (3-Minute Timeframe Version - V1.1 Symmetrical)

This strategy identifies potential mean reversions based on price deviations
from the daily VWAP, filtered by ADX and trend alignment.
It uses an AI model trained specifically for this purpose and adheres to BaseStrategy.
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

# --- Define Internal Filter Parameters (Hardcoded for adherence to BaseStrategy) ---
# These values should ideally match the successful simulation parameters
VWAP_ADX_MIN_THRESH = 20 # Minimum ADX required (avoids chop)
VWAP_ADX_MAX_THRESH = 40 # Maximum ADX allowed (avoids strong trends)
VWAP_TREND_ALIGN_FILTER = True # Whether to use trend alignment
VWAP_TREND_EMA_PERIOD = 50 # EMA period for trend alignment

class VWAP3minStrategy(BaseStrategy):
    """
    VWAP Mean Reversion trading strategy based on AI model predictions,
    incorporating ADX and trend filters, adhering to BaseStrategy definition.
    """

    def __init__(self, model_path: str, scaler_path: str, contract_symbol: str):
        """
        Initialize VWAP Mean Reversion strategy. Parameters match BaseStrategy.

        Args:
            model_path: Path to the VWAP ONNX model (trained with seq_len=40)
            scaler_path: Path to the VWAP scaler
            contract_symbol: Trading symbol (e.g., 'NQ', 'ES')
        """

        super().__init__(model_path, scaler_path, contract_symbol)
        logging.info(f"** Initializing VWAP3minStrategy for {contract_symbol} (3-min TF) **")
        # Store internal filter settings as instance attributes
        self.adx_min_thresh = VWAP_ADX_MIN_THRESH
        self.adx_max_thresh = VWAP_ADX_MAX_THRESH
        self.trend_align_filter = VWAP_TREND_ALIGN_FILTER
        self.trend_ema_period = VWAP_TREND_EMA_PERIOD
        # Log the filter settings being used
        logging.info(f"   Filters: ADX Range=({self.adx_min_thresh}-{self.adx_max_thresh}), TrendAlign={self.trend_align_filter} (EMA{self.trend_ema_period})")


    def get_feature_columns(self) -> List[str]:
        """Returns the 15 feature columns for the VWAP Mean Reversion strategy."""
        # --- Matches feature_cols from training script ---
        return [
            # Momentum Context (6)
            'rsi', 'stoch_k', 'macd_hist', 'rsi_slope', 'stoch_k_slope', 'macd_hist_slope',
            # Dynamic Level Context (4)
            'price_vs_ema20', 'price_vs_ema50', 'price_vs_ema200', 'price_vs_vwap',
            # Trend/Volatility Context (3) - adx is crucial for filtering
            'is_uptrend', 'is_downtrend', 'adx',
            # Mean Reversion Signal Context (2)
            'is_overbought', 'is_oversold'
        ]

    def get_sequence_length(self) -> int:
        """Overrides BaseStrategy: VWAP model uses 40 bars sequence length."""
        # --- Matches seq_len from training script ---
        return 40

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:        
        """
        Adds features required by the VWAP Mean Reversion model and filters.
        Matches the logic in add_vwap_mean_reversion_features from training.
        Includes robust datetime index handling.
        """
        stretch_mult=0.5 # As used in training

        # --- Core Indicators & Momentum ---
        # (These calculations are likely okay, assuming OHLCV data is valid)
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14); df['atr'].replace(0, 1e-6, inplace=True)
        df['ema20'] = ta.ema(df['close'], length=20)
        df[f'ema{self.trend_ema_period}'] = ta.ema(df['close'], length=self.trend_ema_period)
        df['ema200'] = ta.ema(df['close'], length=200)
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=14); df['adx'] = adx_df['ADX_14']
        df['rsi'] = ta.rsi(df['close'], length=14)
        stoch_df = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3, smooth_k=3); df['stoch_k'] = stoch_df['STOCHk_14_3_3']
        macd_df = ta.macd(df['close'], fast=12, slow=26, signal=9); df['macd_hist'] = macd_df['MACDh_12_26_9']
        df['rsi_slope'] = df['rsi'].diff(3); df['stoch_k_slope'] = df['stoch_k'].diff(3); df['macd_hist_slope'] = df['macd_hist'].diff(3)

        # --- Daily Resetting VWAP ---
        # --- FIX: Robust Datetime Index Conversion ---
        if not pd.api.types.is_datetime64_any_dtype(df.index):
             original_index_type = df.index.dtype
             logging.warning(f"Index is not datetime (type: {original_index_type}). Attempting conversion...")
             try:
                 df.index = pd.to_datetime(df.index, errors='coerce')
                 # Check how many NaT values were created
                 nat_count = df.index.isna().sum()
                 if nat_count > 0:
                     logging.warning(f"Found {nat_count} invalid timestamp(s) in index after conversion attempt. Dropping these rows.")
                     # Drop rows where the index conversion failed (became NaT)
                     df = df[df.index.notna()]
                 if df.empty:
                      logging.error("DataFrame became empty after dropping invalid timestamps. Cannot proceed.")
                      # Depending on your bot's logic, you might return an empty df or raise error
                      raise ValueError("DataFrame empty after dropping invalid timestamps.")
                 logging.info("Index successfully converted to datetime.")
             except Exception as e:
                 logging.error(f"CRITICAL: Failed to convert index to datetime for VWAP calc: {e}", exc_info=True)
                 # Re-raise the error to stop processing if conversion fails fundamentally
                 raise ValueError(f"DataFrame index could not be converted to datetime. Check input data format. Error: {e}")
        # --- END FIX ---

        # Proceed only if the index is now definitely datetime
        if not pd.api.types.is_datetime64_any_dtype(df.index):
             # This should ideally not be reached if the above logic is correct
             raise ValueError("Index is still not datetime after conversion attempt. VWAP cannot be calculated.")

        # Calculate VWAP using the datetime index
        dates = pd.Series(df.index.date, index=df.index)
        # Ensure dates align with the potentially modified df index
        if dates.empty or not dates.index.equals(df.index):
             logging.warning("Date series calculation issue post-cleanup. Assigning NaN to VWAP.")
             df['vwap'] = np.nan
        else:
            df['day_id'] = (dates != dates.shift(1)).cumsum()
            df['tp'] = (df['high'] + df['low'] + df['close']) / 3
            df['tpv'] = df['tp'] * df['volume']
            if 'day_id' in df.columns and df['day_id'].nunique() > 0:
                cum_tpv = df.groupby('day_id')['tpv'].cumsum()
                cum_vol = df.groupby('day_id')['volume'].cumsum()
                # Use np.where to handle potential division by zero safely
                df['vwap'] = np.where(cum_vol != 0, cum_tpv / cum_vol, np.nan)
                df.drop(columns=['day_id', 'tp', 'tpv'], inplace=True, errors='ignore')
            else:
                 logging.warning("Could not group by 'day_id' for VWAP calculation. Assigning NaN.")
                 df['vwap'] = np.nan

        # --- Context Features (used by model) ---
        df['is_uptrend'] = (df['close'] > df['ema200']).astype(float)
        df['is_downtrend'] = (df['close'] < df['ema200']).astype(float)
        # Ensure VWAP calculation succeeded before calculating dependent features
        if 'vwap' in df.columns and not df['vwap'].isna().all():
             df['price_vs_vwap'] = (df['close'] - df['vwap']) / df['atr']
             vwap_upper_band = df['vwap'] + (stretch_mult * df['atr'])
             vwap_lower_band = df['vwap'] - (stretch_mult * df['atr'])
             df['is_overbought'] = (df['close'] > vwap_upper_band).astype(float)
             df['is_oversold'] = (df['close'] < vwap_lower_band).astype(float)
        else:
             logging.warning("VWAP column missing or all NaN. Setting VWAP-dependent features to 0.")
             df['price_vs_vwap'] = 0.0
             df['is_overbought'] = 0.0
             df['is_oversold'] = 0.0

        df['price_vs_ema20'] = (df['close'] - df['ema20']) / df['atr']
        df['price_vs_ema50'] = (df['close'] - df[f'ema{self.trend_ema_period}']) / df['atr']
        df['price_vs_ema200'] = (df['close'] - df['ema200']) / df['atr']

        # --- Clean up ---
        # Final fillna after all calculations
        df.replace([np.inf, -np.inf], np.nan, inplace=True);
        # Use more specific fillnas if needed, but ffill/bfill/0 should cover most cases
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        df.fillna(0, inplace=True) # Fill any remaining NaNs (e.g., at the very start) with 0
        logging.debug(f"VWAP features added for {self.contract_symbol}")
        return df


    def load_model(self):
        """Load ONNX model for VWAP Mean Reversion."""
        # --- (Implementation is identical to previous version, logging message updated) ---
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"VWAP Model file not found: {self.model_path}")

            self.model = onnxruntime.InferenceSession(self.model_path)
            logging.info(f"✅ Loaded VWAP model: {os.path.basename(self.model_path)}")
        except Exception as e:
            logging.exception(f"❌ Error loading VWAP model: {e}")
            raise

    def load_scaler(self):
        """Load scaler for VWAP Mean Reversion."""
        # --- (Implementation is identical to previous version, logging message updated) ---
        try:
            if not os.path.exists(self.scaler_path):
                raise FileNotFoundError(f"VWAP Scaler file not found: {self.scaler_path}")

            with open(self.scaler_path, 'rb') as f:
                scalers = pickle.load(f)

            # Extract base ticker (e.g., 'NQ')
            ticker_key = self.contract_symbol.split('.')[-2][:2] if '.' in self.contract_symbol else self.contract_symbol

            if ticker_key in scalers:
                self.scaler = scalers[ticker_key]
                logging.info(f"✅ Loaded '{ticker_key}' scaler for VWAP")
            else:
                available = list(scalers.keys())
                raise ValueError(
                    f"Scaler for '{ticker_key}' (derived from {self.contract_symbol}) not found. "
                    f"Available: {available}"
                )
        except Exception as e:
            logging.exception(f"❌ Error loading VWAP scaler: {e}")
            raise

    def predict(self, df: pd.DataFrame) -> Tuple[int, float]:
        """Generate prediction using VWAP Mean Reversion model."""
        # --- (Implementation uses BaseStrategy.preprocess_features and is identical to previous version) ---
        try:
            features = self.preprocess_features(df) # Handles scaling using self.scaler

            seq_len = self.get_sequence_length() # Gets 40
            if len(features) < seq_len:
                logging.warning(f"Not enough data for VWAP prediction. Need {seq_len}, have {len(features)}. Returning Hold.")
                return 0, 0.0

            X = features[-seq_len:].reshape(1, seq_len, -1).astype(np.float32)

            input_name = self.model.get_inputs()[0].name
            output_name = self.model.get_outputs()[0].name
            logits = self.model.run([output_name], {input_name: X})[0] 

            probs = self._softmax(logits[0])
            prediction = int(np.argmax(probs)) # 0=Hold, 1=Short Win, 2=Long Win
            confidence = float(probs[prediction])

            return prediction, confidence

        except Exception as e:
            logging.exception(f"❌ Prediction error (VWAP): {e}")
            return 0, 0.0

    def should_enter_trade(
        self,
        prediction: int,
        confidence: float,
        bar: Dict, # Requires 'close', 'adx', and the trend EMA (e.g., 'ema50')
        entry_conf: float,
        adx_thresh: float # Per BaseStrategy, this is passed in. We'll use it as the MIN threshold.
    ) -> Tuple[bool, Optional[str]]:
        """
        Determine if VWAP Mean Reversion entry conditions are met, including
        internal ADX range and Trend Alignment filters. Adheres to BaseStrategy signature.
        """
        # --- Step 0: Use the passed adx_thresh as our internal minimum threshold ---
        current_adx_min_thresh = adx_thresh # Use the parameter from the base class signature

        # --- Step 1: Check Confidence Threshold ---
        if confidence < entry_conf:
            logging.debug(f" VWAP Skip: Confidence {confidence:.2f} < {entry_conf}")
            return False, None

        # --- Step 2: Check ADX Range Filter ---
        adx = bar.get('adx', 0) # Get ADX from the latest bar data
        if pd.isna(adx):
             logging.warning(" VWAP Skip: ADX is NaN.")
             return False, None
        # Check MINIMUM ADX (using the passed-in adx_thresh)
        if adx < current_adx_min_thresh:
            logging.debug(f" VWAP Skip: ADX {adx:.1f} < {current_adx_min_thresh} (Chop Filter using adx_thresh param)")
            return False, None
        # Check MAXIMUM ADX (using the internal self.adx_max_thresh)
        if adx > self.adx_max_thresh:
            logging.debug(f" VWAP Skip: ADX {adx:.1f} > {self.adx_max_thresh} (Strong Trend Filter using internal max thresh)")
            return False, None

        # --- Step 3: Check Trend Alignment Filter (If Enabled) ---
        if self.trend_align_filter:
            close_price = bar.get('close', None)
            # Dynamically get the correct EMA column name based on the period
            ema_trend_col = f'ema{self.trend_ema_period}'
            ema_trend = bar.get(ema_trend_col, None)

            if close_price is None or ema_trend is None or pd.isna(ema_trend):
                logging.warning(f" VWAP Skip: Missing data for Trend Alignment Filter (Close or {ema_trend_col}).")
                return False, None

            is_uptrend = close_price > ema_trend
            is_downtrend = close_price < ema_trend

            # Model prediction: 1=Short Win, 2=Long Win
            # Base class expectation: 1=Buy, 2=Sell (We need to map internally)

            # Check Long Signal (Model predicts 2)
            if prediction == 2 and not is_uptrend:
                 logging.debug(f" VWAP Skip: LONG signal (pred=2) blocked by DOWNTREND filter (Close {close_price:.2f} <= {ema_trend_col} {ema_trend:.2f})")
                 return False, None
            # Check Short Signal (Model predicts 1)
            if prediction == 1 and not is_downtrend:
                 logging.debug(f" VWAP Skip: SHORT signal (pred=1) blocked by UPTREND filter (Close {close_price:.2f} >= {ema_trend_col} {ema_trend:.2f})")
                 return False, None

        # --- Step 4: Map Prediction to Trade Direction ---
        # Training labels: 1 = SHORT WIN, 2 = LONG WIN
        if prediction == 1: # Model predicts a successful SHORT
            logging.info(f" VWAP Signal: SHORT (Conf: {confidence:.2f}, ADX: {adx:.1f})")
            return True, 'SHORT'
        elif prediction == 2: # Model predicts a successful LONG
            logging.info(f" VWAP Signal: LONG (Conf: {confidence:.2f}, ADX: {adx:.1f})")
            return True, 'LONG'

        # Prediction is 0 (Hold) or filtered out
        return False, None

    # --- _softmax method is static and utility, no changes needed ---
    @staticmethod
    def _softmax(x):
        """Compute softmax values for array x."""
        if x is None or len(x) == 0:
            return np.array([])
        exp_x = np.exp(x - np.max(x)) # Subtract max for numerical stability
        return exp_x / exp_x.sum()