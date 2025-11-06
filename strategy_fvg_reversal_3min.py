#!/usr/bin/env python3
"""
FVG Reversal Strategy Implementation (V4 - HTF-Aware Transformer)

This strategy uses the final FVG model, which is "aware" of
macro-trend (1H), structure (BoS), and risk (V D) to predict high-probability entries.

VERSION: FINAL-SYNCED - This version is self-sufficient and calculates
all 28 features, including 15-min and 1-hour HTF context,
from the 3-minute data history, perfectly matching the training script.
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
pd.options.mode.chained_assignment = None


class FVGReversal3minStrategy(BaseStrategy):
    """
    FVG Reversal V4.0 (Transformer) trading strategy.
    Trained on 2:1 R:R BoS-Filtered data with 1-hour HTF context.
    """

    def __init__(self, model_path: str, scaler_path: str, contract_symbol: str):
        """
        Initialize FVG Reversal V4.0 strategy.
        """
        super().__init__(model_path, scaler_path, contract_symbol)
        self.BOS_LOOKBACK = 100 
        self.N_PIVOT = 5
        self.EQ_LOOKBACK = 100
        self.EQ_THRESHOLD = 1.0
        
    def get_feature_columns(self) -> List[str]:
        """
        Returns the 28 feature columns for the FVG V4 Transformer model.
        """
        return [
            # 1. Technical (Signal)
            'fvg_signal', 'fvg_age', 'dist_to_fvg_mid', 'close_vs_fvg_mid',
            'is_in_discount', 'is_in_premium', 'market_structure', 
            'is_near_EQH', 'is_near_EQL', 
            'bars_since_BoS', 
            
            # 2. Technical (Context)
            'price_vs_ema200', 'adx', 'price_vel_20', 'body_size', 'wick_ratio', 'atr', 'macro_trend_slope',    
            
            # 3. Contextual (Time)
            'hour_sin', 'hour_cos', 'day_of_week_encoded',  
            
            # 4. Confirmation (Volume/Momentum)
            'mfi', 'volume_roc', 'stoch_rsi',
            
            # 5. V3 Features
            'normalized_bb_width',
            'rejection_wick_ratio',
            
            # 6. ⭐️ V4: HTF (1-Hour) CONTEXT ⭐️
            '1H_market_structure',
            '1H_price_vs_ema',
            '1H_rsi',
        ]

    def get_sequence_length(self) -> int:
        """FVG V4 (Transformer) uses 120 bars."""
        return 120

    # =========================================================
    # FVG V4 FEATURE ENGINEERING (LIVE BOT LOGIC)
    # =========================================================
    
    def _find_fvgs(self, high: np.ndarray, low: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized function to find FVG zones (copied from Script 1)."""
        high_m2 = np.roll(high, 2); low_m2 = np.roll(low, 2)
        bull_fvg = low_m2 > high; bull_top = low_m2; bull_bottom = high
        bear_fvg = high_m2 < low; bear_top = low; bear_bottom = high_m2

        fvg_signal = np.zeros_like(high, dtype=np.int8); fvg_signal[bull_fvg] = 1; fvg_signal[bear_fvg] = -1
        fvg_top = np.where(bull_fvg, bull_top, np.where(bear_fvg, bear_top, np.nan))
        fvg_bottom = np.where(bull_fvg, bull_bottom, np.where(bear_fvg, bear_bottom, np.nan))
        fvg_mid = (fvg_top + fvg_bottom) / 2.0

        fvg_signal[:2] = 0; fvg_top[:2] = np.nan; fvg_mid[:2] = np.nan; fvg_bottom[:2] = np.nan
        return fvg_signal, fvg_top, fvg_mid, fvg_bottom


    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates all 28 features based on raw data input.
        This function is now self-sufficient and generates HTF context
        from the 3-minute data.
        """        
        logging.debug(f"Adding FVG V4 features. Input shape: {df.shape}")
        
        # Ensure the DataFrame has a DatetimeIndex for resampling
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index is not a DatetimeIndex. Cannot perform time-based operations.")

        df = df.copy()

        # === 0. CLEAN OHLCV DATA ===
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(method='ffill').fillna(0)
            else:
                logging.warning(f"Column {col} not found in input DataFrame. Filling with 0.")
                df[col] = 0.0

        # === 1. BASE INDICATORS (3-min) ===
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14).replace(0, 1e-6).fillna(method='ffill').fillna(1e-6)
        df['ema200'] = ta.ema(df['close'], length=200)
        df['price_vs_ema200'] = (df['close'] - df['ema200']) / df['atr']
        df['market_structure'] = np.where(df['close'] > df['ema200'], 1, -1)
        
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=14); df['adx'] = adx_df['ADX_14'].fillna(0) if adx_df is not None else 0
        df['price_vel_20'] = df['close'].diff(20) / df['atr']
        df['body_size'] = abs(df['close'] - df['open']) / df['atr']
        df['wick_ratio'] = (df['high'] - df['low']) / df['atr']
        
        # Volatility Regime (BBW)
        bbands = ta.bbands(df['close'], length=20, std=2)
        if bbands is not None and all(col in bbands.columns for col in ['BBU_20_2.0_2.0', 'BBL_20_2.0_2.0', 'BBM_20_2.0_2.0']):
            df['normalized_bb_width'] = (bbands['BBU_20_2.0_2.0'] - bbands['BBL_20_2.0_2.0']) / (bbands['BBM_20_2.0_2.0'] + 1e-6)
        else:
            df['normalized_bb_width'] = 0.0
            logging.warning("Bollinger Band calculation failed. Filling 'normalized_bb_width' with 0.")
        
        # Rejection Wick
        df['rejection_wick_ratio'] = np.where(df['close'] > df['open'],
                                            (df['close'] - df['low']) / df['atr'],
                                            (df['high'] - df['close']) / df['atr'])

        # --- 2. FVG SIGNAL FEATURES ---
        fvg_signal, fvg_top, fvg_mid, fvg_bottom = self._find_fvgs(df['high'].values, df['low'].values)
        
        df['fvg_signal_raw'] = fvg_signal # Store raw signal for age calculation
        
        df_fvg = pd.DataFrame({'fvg_signal_raw': fvg_signal, 'fvg_top': fvg_top, 'fvg_mid': fvg_mid, 'fvg_bottom': fvg_bottom}, index=df.index).replace(0, np.nan).ffill(limit=self.BOS_LOOKBACK).fillna(0)
        
        df['fvg_signal'] = df_fvg['fvg_signal_raw']
        df['fvg_mid'] = df_fvg['fvg_mid']
        
        df['fvg_age'] = df.groupby((df['fvg_signal_raw'] != 0).cumsum()).cumcount()
        
        df['dist_to_fvg_mid'] = (df['fvg_mid'] - df['close']) / df['atr']
        df.loc[df['fvg_signal'] == 0, 'dist_to_fvg_mid'] = 0
        df['close_vs_fvg_mid'] = (df['close'] - df['fvg_mid']) / df['atr']
        df.loc[df['fvg_signal'] == 0, 'close_vs_fvg_mid'] = 0

        # Premium/Discount
        swing_range_high = df['high'].rolling(50).max(); swing_range_low = df['low'].rolling(50).min()
        swing_range_mid = (swing_range_high + swing_range_low) / 2.0
        df['is_in_discount'] = np.where(df['close'] < swing_range_mid, 1, 0)
        df['is_in_premium'] = np.where(df['close'] > swing_range_mid, 1, 0)
        
        # BoS Age (BoS_long/short and bars_since_BoS must be calculated)
        is_sh = (df['high'] == df['high'].rolling(self.N_PIVOT * 2 + 1, center=True).max()).shift(self.N_PIVOT).fillna(False)
        is_sl = (df['low'] == df['low'].rolling(self.N_PIVOT * 2 + 1, center=True).min()).shift(self.N_PIVOT).fillna(False)

        sh_val = df['high'].where(is_sh).ffill(); sl_val = df['low'].where(is_sl).ffill()
        
        df['BoS_long_raw'] = (df['high'] > sh_val.shift(1)).astype(int) 
        df['BoS_short_raw'] = (df['low'] < sl_val.shift(1)).astype(int)
        
        last_bos_long_idx = df.index.to_series().where(df['BoS_long_raw'] == 1).ffill()
        last_bos_short_idx = df.index.to_series().where(df['BoS_short_raw'] == 1).ffill()

        df['bars_since_BoS_long'] = df.index.get_indexer(df.index) - df.index.get_indexer(last_bos_long_idx)
        df['bars_since_BoS_short'] = df.index.get_indexer(df.index) - df.index.get_indexer(last_bos_short_idx)

        df['bars_since_BoS_long'].fillna(self.BOS_LOOKBACK + 1, inplace=True)
        df['bars_since_BoS_short'].fillna(self.BOS_LOOKBACK + 1, inplace=True)
        
        # Select the correct BoS age based on local market structure
        df['bars_since_BoS'] = np.where(df['market_structure'] == 1, 
                                     df['bars_since_BoS_long'], 
                                     df['bars_since_BoS_short'])

        # EQH/EQL
        df['is_near_EQH'] = (df['high'].rolling(self.EQ_LOOKBACK).max().shift(1) - df['high']).abs() < self.EQ_THRESHOLD
        df['is_near_EQL'] = (df['low'] - df['low'].rolling(self.EQ_LOOKBACK).min().shift(1)).abs() < self.EQ_THRESHOLD
        df['is_near_EQH'] = df['is_near_EQH'].astype(int)
        df['is_near_EQL'] = df['is_near_EQL'].astype(int)
        
        # --- 3. Confirmation and Context ---
        mfi_df = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=14); df['mfi'] = mfi_df.fillna(50)
        df['volume_roc'] = ta.roc(df['volume'], length=5).fillna(0); stochrsi_df = ta.stochrsi(df['close'], length=14); df['stoch_rsi'] = stochrsi_df['STOCHRSIk_14_14_3_3'].fillna(0)

        hours_in_day = 24; df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / hours_in_day); df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / hours_in_day); df['day_of_week_encoded'] = df.index.dayofweek / 6.0
        
        # --- 4. ⭐️ HTF CONTEXT (Resampled from 3-min) ⭐️ ---
        
        # 15-min Macro Slope
        df_15min = df[['open', 'high', 'low', 'close', 'volume']].resample('15Min', label='left', closed='left').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        if not df_15min.empty:
            ema_15m_series = ta.ema(df_15min['close'], length=40)
            df_15m_features = pd.DataFrame(index=df_15min.index)
            df_15m_features['ema40_slope'] = ema_15m_series.diff(3)
            df_15m_features = df_15m_features.shift(1) # Prevent lookahead
            df = pd.merge_asof(df, df_15m_features, left_index=True, right_index=True, direction='backward')
            df['macro_trend_slope'] = df['ema40_slope'] / df['atr'] # Normalized
        else:
            df['macro_trend_slope'] = 0.0

        # 1-Hour HTF Context
        agg_dict = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
        df_1hour = df[['open', 'high', 'low', 'close', 'volume']].resample('1H', label='left', closed='left').agg(agg_dict).dropna()
        
        if not df_1hour.empty:
            df_1h_features = pd.DataFrame(index=df_1hour.index)
            df_1h_features['1H_atr'] = ta.atr(df_1hour['high'], df_1hour['low'], df_1hour['close'], length=14).replace(0, 1e-6)
            df_1h_features['1H_ema20'] = ta.ema(df_1hour['close'], length=20)
            df_1h_features['1H_ema50'] = ta.ema(df_1hour['close'], length=50)
            df_1h_features['1H_ema200'] = ta.ema(df_1hour['close'], length=200)
            df_1h_features['1H_rsi'] = ta.rsi(df_1hour['close'], length=14).fillna(50)
            
            df_1h_features['1H_market_structure'] = (df_1h_features['1H_ema50'] - df_1h_features['1H_ema200']) / df_1h_features['1H_atr']
            df_1h_features['1H_price_vs_ema'] = (df_1hour['close'] - df_1h_features['1H_ema20']) / df_1h_features['1H_atr']
            
            df_1h_features = df_1h_features.shift(1) # Prevent lookahead
            
            df = pd.merge_asof(df, df_1h_features[['1H_market_structure', '1H_price_vs_ema', '1H_rsi']], 
                                  left_index=True, right_index=True, direction='backward')
        else:
            df['1H_market_structure'] = 0.0
            df['1H_price_vs_ema'] = 0.0
            df['1H_rsi'] = 50.0
        
        # === FINAL CLEANUP AND FEATURE RETURN ===
        all_feature_cols = self.get_feature_columns()
        
        # Fill any missing HTF/BoS/Macro columns with zeros if the backtest runner didn't merge them.
        for col in all_feature_cols:
            if col not in df.columns: 
                logging.warning(f"Feature {col} missing in execution DF. Filling with zeros.")
                df[col] = 0.0
                
        # Fill NaNs created during calculations
        df[all_feature_cols] = df[all_feature_cols].replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)

        logging.debug(f"FVG V4 features added. Shape after features: {df.shape}")
        
        # ⭐️⭐️⭐️ CRITICAL FIX: Return the *entire* DataFrame ⭐️⭐️⭐️
        # The base bot needs the OHLCV columns for execution.
        # The preprocess_features function will handle selecting the 28 feature columns.
        return df


    # =========================================================
    # LIVE EXECUTION FUNCTIONS (Unchanged)
    # =========================================================

    def load_model(self):
        """Load ONNX model for FVG Reversal."""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            self.model = onnxruntime.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
            logging.info(f"✅ Loaded FVG Reversal V4 model: {os.path.basename(self.model_path)}")
        except Exception as e:
            logging.exception(f"❌ Error loading FVG Reversal V4 model: {e}")
            raise

    def load_scaler(self):
        """Load scaler for FVG Reversal."""
        try:
            if not os.path.exists(self.scaler_path):
                raise FileNotFoundError(f"Scaler file not found: {self.scaler_path}")

            with open(self.scaler_path, 'rb') as f:
                scalers = pickle.load(f)

            base_symbol = self.contract_symbol.split('.')[0][:2] 
            
            if base_symbol in scalers:
                self.scaler = scalers[base_symbol]
                logging.info(f"✅ Loaded '{base_symbol}' scaler for FVG Reversal V4")
            else:
                available = list(scalers.keys())
                raise ValueError(
                    f"'{base_symbol}' scaler not found. "
                    f"Available: {available}"
                )
        except Exception as e:
            logging.exception(f"❌ Error loading FVG Reversal V4 scaler: {e}")
            raise

    def predict(self, df: pd.DataFrame) -> Tuple[int, float]:
        """
        Generate prediction using FVG V4 Transformer model.
        """
        try:
            # preprocess_features is defined in the BaseStrategy.
            # It calls add_features(), then scales, then returns a NumPy array.
            features_array = self.preprocess_features(df) 

            seq_len = self.get_sequence_length() 
            if len(features_array) < seq_len:
                logging.warning(f"⚠️ Not enough data for prediction. Need {seq_len}, have {len(features_array)}. Returning Hold.")
                return 0, 0.0

            # ⭐️ FIX: features_array is already a NumPy array, do not call .values again.
            X = features_array[-seq_len:].reshape(1, seq_len, -1).astype(np.float32)

            input_name = self.model.get_inputs()[0].name
            output_name = self.model.get_outputs()[0].name
            logits_sequence = self.model.run([output_name], {input_name: X})[0] 

            last_logits = logits_sequence[0, -1, :] 

            probs = self._softmax(last_logits)
            prediction = int(np.argmax(probs))
            confidence = float(probs[prediction])

            return prediction, confidence

        except Exception as e:
            logging.exception(f"❌ Prediction error (FVG Reversal V4): {e}")
            return 0, 0.0 

    def should_enter_trade(
        self,
        prediction: int,
        confidence: float,
        bar: Dict,
        entry_conf: float,
        adx_thresh: float 
    ) -> Tuple[bool, Optional[str]]:
        """
        Determine if FVG Reversal V4 entry conditions are met.
        """
        
        if confidence < entry_conf:
            return False, None
        
        # FVG V4 model uses: 1=BUY (Long), 2=SELL (Short)
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