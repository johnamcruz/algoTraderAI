#!/usr/bin/env python3
"""
Supertrend Pullback Strategy Implementation (V4.0 - Transformer)
This strategy is based on the final, risk-hardened V4.0 configuration (28 Features)
and is built for ONNX runtime deployment.
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


class SupertrendPullbackStrategy2(BaseStrategy):
    """
    Supertrend Pullback V4.0 (Transformer) trading strategy.
    Uses "Pure AI Mode" with risk-hardened features.
    """

    def __init__(self, model_path: str, scaler_path: str, contract_symbol: str):
        """
        Initialize SupertrendPullbackStrategy (V4.0)
        """
        super().__init__(model_path, scaler_path, contract_symbol)
        logging.info("Initialized SupertrendPullbackStrategy (V4.0 - 28 Features - MTF Hardened)")
        # Store long-term MTF history (critical for causal processing)
        self.mtf_history: Dict[str, pd.DataFrame] = {}


    def get_feature_columns(self) -> List[str]:
        """
        Returns the 28 feature columns for the final V4.0 Supertrend Transformer.
        """
        # ⭐️ V4.0 FEATURE COLS: V3.10 CORE FEATURES + MTF (Total 28) ⭐️
        return [
            'price_vs_st', 'st_direction', 'st_val_slow', 'st_direction_slow', 'price_vs_st_slow',
            'price_vs_ema40', 'ema15_vs_ema40', 'price_vs_ema200', 'adx',
            'price_vel_20',
            'body_size', 'wick_ratio', 'atr',
            'macro_trend_slope', 'price_roc_slope', 'inverse_volatility_score', 'volume_velocity', # V3.10 Restored Core
            'st_slope_long', 'dist_to_ema200', 'adx_acceleration_5', # Contextual
            'mfi', 'volume_roc', 'stoch_rsi', # Volume/Momentum
            'hour_sin', 'hour_cos', 'day_of_week_encoded', # Time
            'normalized_bb_width', # Volatility (Kept V3.11 Feature)
            'macro_trend_slope_60m' # NEW 60M MTF FEATURE
        ]

    def get_sequence_length(self) -> int:
        """
        Supertrend Pullback V4.0 (Transformer) uses 120 bars.
        """
        return 120

    # =========================================================
    # V4.0 FEATURE ENGINEERING (Restored V3.10 + MTF)
    # =========================================================
    
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate V4.0 features, including MTF trends.
        """        
        logging.debug(f"Adding V4.0 Supertrend features. Input shape: {df.shape}")
        
        df = df.copy()
        
        # --- 1. Core Indicators (3-min TF) ---
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14).replace(0, 1e-6)
        df['ema15'] = ta.ema(df['close'], length=15); df['ema40'] = ta.ema(df['close'], length=40); df['ema200'] = ta.ema(df['close'], length=200)
        st_df_fast = ta.supertrend(df['high'], df['low'], df['close'], length=10, multiplier=3)
        df['st_val'] = st_df_fast['SUPERT_10_3']; df['st_direction'] = st_df_fast['SUPERTd_10_3']
        st_df_slow = ta.supertrend(df['high'], df['low'], df['close'], length=20, multiplier=4)
        df['st_val_slow'] = st_df_slow['SUPERT_20_4']; df['st_direction_slow'] = st_df_slow['SUPERTd_20_4']
        
        # --- 2. Normalized Price/Momentum Features (3-min TF) ---
        df['price_vs_st'] = (df['close'] - df['st_val']) / df['atr'];
        df['price_vs_st_slow'] = (df['close'] - df['st_val_slow']) / df['atr'];
        df['price_vs_ema40'] = (df['close'] - df['ema40']) / df['atr'];
        df['ema15_vs_ema40'] = (df['ema15'] - df['ema40']) / df['atr']
        df['price_vs_ema200'] = (df['close'] - df['ema200']) / df['atr']
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
        df['adx'] = adx_df['ADX_14'].fillna(0)
        
        # Intermediate calculation
        df['adx_slope_calc'] = df['adx'].diff(5) 
        df['st_slope_long'] = df['st_val_slow'].diff(50) / (df['atr'] * 50)                
        df['dist_to_ema200'] = abs(df['close'] - df['ema200']) / df['atr']        
        df['adx_acceleration_5'] = df['adx_slope_calc'].diff(5)
                
        df['price_vel_20'] = df['close'].diff(20) / df['atr'];         
        df['body_size'] = abs(df['close'] - df['open']) / df['atr']; df['wick_ratio'] = (df['high'] - df['low']) / df['atr']

        # --- 3. V3.10 Restored Features (Volatility and Volume Velocity) ---
        df['vol_channel_width'] = (df['ema15'] - df['ema40']).abs()
        df['inverse_volatility_score'] = df['vol_channel_width'] / (df['atr'] + 1e-6)
        vol_std = df['volume'].rolling(10).std().replace(0, 1e-6); df['volume_velocity'] = df['volume'].diff(1) / vol_std
        df['price_vel_10_calc'] = df['close'].diff(10) / df['atr'] 
        df['price_roc_slope'] = df['price_vel_10_calc'].diff(10)

        # V3.11 Feature Kept (Normalized BB Width)
        bbands = ta.bbands(df['close'], length=20, std=2)
        df['normalized_bb_width'] = (bbands['BBU_20_2.0_2.0'] - bbands['BBL_20_2.0_2.0']) / (df['atr'] * 2 + 1e-6).fillna(0.0)

        # Volume/Momentum/Time features
        mfi_df = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=14)
        df['mfi'] = mfi_df.fillna(50) 
        df['volume_roc'] = ta.roc(df['volume'], length=5).fillna(0)
        stochrsi_df = ta.stochrsi(df['close'], length=14)
        df['stoch_rsi'] = stochrsi_df['STOCHRSIk_14_14_3_3'].fillna(0)
        
        # Time-Based Contextual Features (Cyclical)
        df['hour'] = df.index.hour
        hours_in_day = 24
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / hours_in_day)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / hours_in_day)
        df['day_of_week'] = df.index.dayofweek
        df['day_of_week_encoded'] = df['day_of_week'] / 6.0 

        # --- 4. MTF Feature Calculation and Merge (15M and 60M) ---
        
        # Helper function for MTF resampling and EMA slope calculation
        def calculate_mtf_slope(df_base: pd.DataFrame, timeframe: str):
            """Resamples price data and calculates 40 EMA slope robustly."""
            
            # 1. Resample and Check for valid data
            df_mtf = df_base['close'].resample(timeframe, label='left', closed='left').ohlc()

            # Create a base DataFrame with the original index structure for safe merging
            df_mtf_features = pd.DataFrame(index=df_mtf.index)
            
            if df_mtf.empty or 'close' not in df_mtf.columns or df_mtf['close'].isnull().all():
                logging.warning(f"⚠️ MTF calculation failed for {timeframe}: Insufficient or invalid data.")
                # Return an empty DataFrame that *should* merge harmlessly
                return pd.DataFrame(columns=['ema40_slope', 'macro_trend_direction_60m'], index=df_base.index.copy())
            
            # 2. Calculate 40 EMA
            # Use dropna() on the series itself, not the DataFrame index
            ema_series = ta.ema(df_mtf['close'].dropna(), length=40)
            
            if ema_series is None or ema_series.empty or ema_series.isnull().all():
                logging.warning(f"⚠️ MTF calculation failed for {timeframe}: EMA calculation returned invalid data.")
                return pd.DataFrame(columns=['ema40_slope', 'macro_trend_direction_60m'], index=df_base.index.copy())

            # 3. Calculate Slope and Direction
            
            # Use the index from the calculated series
            df_mtf_features = pd.DataFrame(index=ema_series.index)
            
            df_mtf_features['ema40_slope'] = ema_series.diff(3)
            
            # This step will be done during the final MTF calculation using df_mtf_features['ema40_slope']
            # df_mtf_features['macro_trend_direction'] = np.where(df_mtf_features['ema40_slope'] > 0, 1, -1)
            
            # Shift back to prevent lookahead and drop initial NaNs caused by the rolling window
            df_mtf_features = df_mtf_features.shift(1).dropna().copy()
            
            # Apply the direction logic only now that the data is shifted and clean
            df_mtf_features[f'ema40_direction'] = np.where(df_mtf_features['ema40_slope'] > 0, 1, -1) 
            
            # Rename columns for the 60M merge if necessary, otherwise use the standard names.
            if timeframe == '60Min':
                df_mtf_features = df_mtf_features.rename(columns={
                    'ema40_slope': 'ema40_slope_60m', 
                    'ema40_direction': 'macro_trend_direction_60m'
                })
                
            return df_mtf_features

        # 4a. 15M Tactical Trend (Original V3.10 MTF)
        df_15m_features = calculate_mtf_slope(df, '15Min')
        df = pd.merge_asof(df, df_15m_features, left_index=True, right_index=True)
        df['macro_trend_slope'] = df['ema40_slope_15Min'] / df['atr'] if 'ema40_slope_15Min' in df.columns else 0.0

        # 4b. 60M Strategic Trend (New V4.0 MTF)
        df_60m_features = calculate_mtf_slope(df, '60Min')
        df = pd.merge_asof(df, df_60m_features, left_index=True, right_index=True, suffixes=('_15m', '_60m'))
        df['macro_trend_slope_60m'] = df['ema40_slope_60m'] / df['atr'] if 'ema40_slope_60m' in df.columns else 0.0


        # --- 5. Final Cleanup ---
        
        # Intermediate columns to drop
        intermediate_cols = [
            'hour', 'day_of_week', 'adx_slope_calc', 'vol_channel_width', 'price_vel_10_calc',
            'ema40_slope_15Min', 'ema40_slope_60m' # Drop intermediate MTF slopes
        ]

        df.drop(columns=intermediate_cols, inplace=True, errors='ignore')                
        
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Use a larger ffill to ensure no NaNs from rolling/diff features
        df.fillna(method='ffill', limit=200, inplace=True) 
        df.fillna(method='bfill', limit=200, inplace=True) 
        df.fillna(0, inplace=True)

        all_feature_cols = self.get_feature_columns()
        for col in all_feature_cols:
            if col not in df.columns: 
                df[col] = 0.0
                    
        logging.debug(f"V4.0 Supertrend features added. Shape after features: {df.shape}")
        return df

    # =========================================================
    # LIVE EXECUTION FUNCTIONS (ONNX)
    # =========================================================

    def load_model(self):
        """Load ONNX model for Supertrend Pullback (V4.0)."""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            self.model = onnxruntime.InferenceSession(self.model_path)
            logging.info(f"✅ Loaded Supertrend Pullback V4.0 ONNX model: {os.path.basename(self.model_path)}")
        except Exception as e:
            logging.exception(f"❌ Error loading Supertrend Pullback V4.0 model: {e}")
            raise

    def load_scaler(self):
        """Load scaler for Supertrend Pullback (V4.0)."""
        try:
            if not os.path.exists(self.scaler_path):
                raise FileNotFoundError(f"Scaler file not found: {self.scaler_path}")

            with open(self.scaler_path, 'rb') as f:
                scalers = pickle.load(f)

            base_symbol = self.contract_symbol.split('.')[0][:2] 
            
            if base_symbol in scalers:
                self.scaler = scalers[base_symbol]
                logging.info(f"✅ Loaded '{base_symbol}' scaler for Supertrend V4.0")
            else:
                available = list(scalers.keys())
                raise ValueError(
                    f"'{base_symbol}' scaler not found. "
                    f"Available: {available}"
                )
        except Exception as e:
            logging.exception(f"❌ Error loading Supertrend V4.0 scaler: {e}")
            raise

    # --- (No change to predict/should_enter_trade functions, they are correct) ---
    def predict(self, df: pd.DataFrame) -> Tuple[int, float]:
        """
        Generate prediction using V4.0 Transformer model (ONNX).
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
            logging.exception(f"❌ Prediction error (Supertrend V4.0): {e}")
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
        Determine if V4.0 "Pure AI Mode" entry conditions are met.
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