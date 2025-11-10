#!/usr/bin/env python3
"""
Supertrend Pullback Strategy Implementation (V4.8 - Transformer)
This strategy is based on the final, outcome-weighted V4.8 configuration (32 Features)
Hardened version to prevent feature dropping during calculation and ensures only 32
features are returned for prediction (OHLCV is explicitly dropped).
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
    Supertrend Pullback V4.8 (Transformer) trading strategy.
    Hardened feature engineering for production stability.
    """

    def __init__(self, model_path: str, scaler_path: str, contract_symbol: str):
        super().__init__(model_path, scaler_path, contract_symbol)
        logging.info("Initialized SupertrendPullbackStrategy (V4.8 - 32 Features - Hardened)")
        self.mtf_history: Dict[str, pd.DataFrame] = {}


    def get_feature_columns(self) -> List[str]:
        """
        Returns the 32 feature columns for the final V4.8 Supertrend Transformer.
        """
        # ⭐️ V4.8 FEATURE COLS: 32 FEATURES ⭐️
        return [
            # TIER 1: CORE TREND & MTF (12)
            'st_direction', 'st_direction_slow', 'price_vs_st', 'price_vs_st_slow',
            'adx', 'price_vs_ema40', 'ema15_vs_ema40', 'price_vs_ema200',
            'mtf_alignment_score', 'macro_trend_slope', 'macro_trend_direction_60m', 'ema40_direction',

            # TIER 2: PULLBACK & REGIME QUALITY (6)
            'tradeable_trend', 'pullback_orderly', 'fresh_structure',
            'ema_separation', 'volatility_regime', 'volatility_spike',

            # TIER 3: MOMENTUM & VOLUME (6)
            'price_roc_slope', 'volume_thrust', 'volume_declining',
            'stoch_rsi', 'mfi', 'adx_acceleration_5',

            # TIER 4: CANDLE & RISK (5)
            'atr', 'body_size', 'normalized_bb_width', 'hour_sin', 'hour_cos',

            # TIER 5: CONTEXT & RISK (3)
            'day_of_week_encoded', 'setup_quality_score', 'filter_strength'
        ]

    def get_sequence_length(self) -> int:
        return 120

    # =========================================================
    # V4.8 FEATURE ENGINEERING (32 FEATURES) - HARDENED
    # =========================================================
    
    def _calculate_mtf_slope_hardened(self, df_base: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Robustly resamples price data and calculates 40 EMA slope."""
        
        # 1. Resample and Check for valid data
        try:
            df_mtf = df_base['close'].resample(timeframe, label='left', closed='left').ohlc()
        except Exception as e:
            logging.warning(f"⚠️ MTF Resample failed for {timeframe}: {e}")
            return pd.DataFrame()

        df_mtf_features = pd.DataFrame(index=df_mtf.index)
        
        if df_mtf.empty or 'close' not in df_mtf.columns or df_mtf['close'].isnull().all():
            logging.warning(f"⚠️ MTF calculation failed for {timeframe}: Insufficient or invalid data.")
            return pd.DataFrame()
        
        # 2. Calculate 40 EMA and Slope
        ema_series = ta.ema(df_mtf['close'].dropna(), length=40)
        
        if ema_series is None or ema_series.empty or ema_series.isnull().all():
            logging.warning(f"⚠️ MTF calculation failed for {timeframe}: EMA calculation returned invalid data.")
            return pd.DataFrame()

        df_mtf_features = pd.DataFrame(index=ema_series.index)
        # Use a prefixed name for the slope intermediate column
        slope_col = f'_calc_ema40_slope_{timeframe.lower().replace("min", "m")}'
        dir_col = 'ema40_direction' if timeframe != '60Min' else 'macro_trend_direction_60m'
        
        df_mtf_features[slope_col] = ema_series.diff(3)
        
        # Shift back to prevent lookahead and drop initial NaNs
        df_mtf_features = df_mtf_features.shift(1).dropna().copy()
        
        # Apply the direction logic
        df_mtf_features[dir_col] = np.where(df_mtf_features[slope_col] > 0, 1, -1) 
                
        return df_mtf_features


    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate V4.8 features with robust error handling and cleanup,
        ensuring OHLCV data is used for calculation but dropped before return.
        """        
        logging.debug(f"Adding V4.8 Supertrend features. Input shape: {df.shape}")
        
        # ⭐️ CRITICAL: Operate on a copy to ensure input OHLCV is preserved for calculation
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
        
        # Intermediate calculations (prefixed for later cleanup)
        df['_calc_adx_slope'] = df['adx'].diff(5) 
        df['adx_acceleration_5'] = df['_calc_adx_slope'].diff(5)
                
        df['_calc_price_vel_20'] = df['close'].diff(20) / df['atr'];
        df['price_roc_slope'] = df['_calc_price_vel_20'].diff(10) # Price ROC slope from 20-bar velocity
                 
        df['body_size'] = abs(df['close'] - df['open']) / df['atr']; 
        df['wick_ratio'] = (df['high'] - df['low']) / df['atr']

        # Volatility & Volume Features
        bbands = ta.bbands(df['close'], length=20, std=2)
        df['normalized_bb_width'] = (bbands['BBU_20_2.0_2.0'] - bbands['BBL_20_2.0_2.0']) / (df['atr'] * 2 + 1e-6)
        df['normalized_bb_width'] = df['normalized_bb_width'].fillna(0.0)

        mfi_df = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=14)
        df['mfi'] = mfi_df.fillna(50) 
        df['volume_roc'] = ta.roc(df['volume'], length=5).fillna(0)
        
        stochrsi_df = ta.stochrsi(df['close'], length=14)
        df['stoch_rsi'] = stochrsi_df['STOCHRSIk_14_14_3_3'].fillna(0)
        
        # Volume features for V4.8
        df['_calc_volume_ma20'] = df['volume'].rolling(20).mean()
        df['volume_thrust'] = df['volume'] / (df['_calc_volume_ma20'] + 1e-6)
        df['volume_declining'] = (df['volume'].rolling(3).mean() < df['volume'].rolling(10).mean()).astype(int)

        # Time-Based Contextual Features (Cyclical)
        df['_calc_hour'] = df.index.hour
        hours_in_day = 24
        df['hour_sin'] = np.sin(2 * np.pi * df['_calc_hour'] / hours_in_day)
        df['hour_cos'] = np.cos(2 * np.pi * df['_calc_hour'] / hours_in_day)
        df['_calc_day_of_week'] = df.index.dayofweek
        df['day_of_week_encoded'] = df['_calc_day_of_week'] / 6.0 

        # --- 3. V4.8 Pullback & Regime Quality Features ---
        
        # Regime/Trend
        df['_calc_st_consistency_20'] = df['st_direction'].rolling(20).sum() / 20
        df['tradeable_trend'] = (df['_calc_st_consistency_20'].abs() >= 0.7).astype(int)
        df['ema_separation'] = (df['ema15'] - df['ema40']).abs() / df['atr']

        df['_calc_atr_ma20'] = df['atr'].rolling(20).mean()
        df['volatility_regime'] = df['atr'] / (df['_calc_atr_ma20'] + 1e-6)
        df['volatility_spike'] = (df['volatility_regime'] > 1.25).astype(int)

        # Pullback Quality
        df['pullback_orderly'] = (df['body_size'].rolling(3).std() < df['body_size'].rolling(20).mean() * 0.5).astype(int)
        
        # Structure
        df['_calc_swing_high'] = ((df['high'] > df['high'].shift(1)) &
                             (df['high'] > df['high'].shift(-1))).astype(int)
        df['_calc_swing_low'] = ((df['low'] < df['low'].shift(1)) &
                            (df['low'] < df['low'].shift(-1))).astype(int)
        swing_marker = (df['_calc_swing_high'] | df['_calc_swing_low']).astype(bool)
        df['_calc_bars_since_swing'] = df.groupby(swing_marker.cumsum()).cumcount()
        df['fresh_structure'] = (df['_calc_bars_since_swing'] <= 10).astype(int)

        # Rejection Wicks (Intermediate for setup_quality_score)
        df['_calc_upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['_calc_lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['_calc_rejection_wick_long'] = (df['_calc_lower_wick'] / (df['atr'] + 1e-6) > 0.5).astype(int)
        df['_calc_rejection_wick_short'] = (df['_calc_upper_wick'] / (df['atr'] + 1e-6) > 0.5).astype(int)
        
        # --- 4. MTF Feature Calculation and Merge (15M and 60M) ---
        
        # Initialize default values for MTF-related features
        df['ema40_direction'] = 0.0 # 15m direction
        df['macro_trend_direction_60m'] = 0.0 # 60m direction
        df['macro_trend_slope'] = 0.0 # 15m slope normalized
        
        # 4a. 15M Trend
        df_15m_features = self._calculate_mtf_slope_hardened(df, '15Min')
        
        # Merge only the EMA Direction feature
        if 'ema40_direction' in df_15m_features.columns:
            # Temporarily rename for merge and drop if necessary
            df_15m_direction = df_15m_features[['ema40_direction']].copy()
            df_15m_direction.rename(columns={'ema40_direction': '_calc_ema40_direction_15m'}, inplace=True)
            df = pd.merge_asof(df, df_15m_direction, left_index=True, right_index=True, direction='forward')
            
            # Update the official feature column and clean up the temporary one
            df['ema40_direction'] = df['_calc_ema40_direction_15m'].fillna(0.0)
            df.drop(columns=['_calc_ema40_direction_15m'], inplace=True, errors='ignore')

        # Calculate macro_trend_slope (requires the slope from the helper output)
        _15m_slope_col = f'_calc_ema40_slope_15m'
        if _15m_slope_col in df_15m_features.columns:
             # Merge the slope separately, as it's only needed for calculation
            df_15m_slope = df_15m_features[[_15m_slope_col]].copy()
            df = pd.merge_asof(df, df_15m_slope, left_index=True, right_index=True, direction='forward')
            df['macro_trend_slope'] = df[_15m_slope_col] / df['atr']
            df.drop(columns=[_15m_slope_col], inplace=True, errors='ignore') # Drop the merged slope column
        else:
            df['macro_trend_slope'] = 0.0
            
        # 4b. 60M Strategic Trend
        df_60m_features = self._calculate_mtf_slope_hardened(df, '60Min')
        
        if 'macro_trend_direction_60m' in df_60m_features.columns:
            # Merge the 60m direction feature
            df_60m_direction = df_60m_features[['macro_trend_direction_60m']].copy()
            df_60m_direction.rename(columns={'macro_trend_direction_60m': '_calc_macro_direction_60m'}, inplace=True)
            df = pd.merge_asof(df, df_60m_direction, left_index=True, right_index=True, direction='forward')

            # Update the official feature column and clean up the temporary one
            df['macro_trend_direction_60m'] = df['_calc_macro_direction_60m'].fillna(0.0)
            df.drop(columns=['_calc_macro_direction_60m'], inplace=True, errors='ignore')
            
            # Also drop the merged slope column if it exists in the 60m features
            _60m_slope_col = f'_calc_ema40_slope_60m'
            if _60m_slope_col in df_60m_features.columns:
                 df.drop(columns=[_60m_slope_col], inplace=True, errors='ignore')

        # MTF Alignment Score (V4.8)
        df['mtf_alignment_score'] = (
            df['st_direction'] +
            df['ema40_direction'] +
            df['macro_trend_direction_60m']
        )
        
        # --- 5. Composite Quality Scores (V4.8) ---
        
        wick_rejection = (df['_calc_rejection_wick_long'] | df['_calc_rejection_wick_short']).astype(int)
        
        df['setup_quality_score'] = (
            df['tradeable_trend'] * 0.25 +
            df['pullback_orderly'] * 0.20 +
            df['volume_declining'] * 0.15 +
            wick_rejection * 0.20 +
            df['fresh_structure'] * 0.20
        )

        df['filter_strength'] = (
            (np.abs(df['close'] - df['st_val']) / df['atr'] < 2.0).astype(float) +
            (df['adx'] > 20).astype(float) +
            (df['setup_quality_score'] > 0.5).astype(float) +
            (df['volume_declining'] == 1).astype(float)
        ) / 4.0
        

        # --- 6. Drop Intermediate Columns ---
        
        # Drop all prefixed intermediate columns and other temporary indicators
        intermediate_cols_to_drop = [
            col for col in df.columns if col.startswith('_calc_') or col in [
                'st_val', 'st_val_slow', 'ema15', 'ema40', 'ema200', # Non-feature indicators
            ]
        ]
        
        df.drop(columns=intermediate_cols_to_drop, inplace=True, errors='ignore')                
        
        # Robust NaN/Inf Handling
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(method='ffill', limit=200, inplace=True) 
        df.fillna(method='bfill', limit=200, inplace=True) 
        df.fillna(0, inplace=True)

        # --- 7. EXPLICITLY DROP OHLCV & Other Non-Features ---
        final_feature_cols = self.get_feature_columns()
        
        # The list of non-feature columns that MUST NOT go to the model
        cols_to_keep = set(final_feature_cols)
        cols_to_drop = [col for col in df.columns if col not in cols_to_keep]

        df.drop(columns=cols_to_drop, inplace=True, errors='ignore')


        # --- 8. Final Verification of ALL 32 Features ---
        missing_cols = []
        for col in final_feature_cols:
            if col not in df.columns: 
                df[col] = 0.0 # Add the missing column back as zeros
                missing_cols.append(col)

        if missing_cols:
            logging.warning(f"⚠️ Added missing V4.8 features as zeros (likely due to insufficient history): {', '.join(missing_cols)}")
                    
        logging.debug(f"V4.8 Supertrend features added. Shape after features: {df.shape}")
        
        # Ensure the final DataFrame order matches the required feature order
        df = df[final_feature_cols] 
        
        return df

    # =========================================================
    # LIVE EXECUTION FUNCTIONS (ONNX) - No Change Needed
    # =========================================================

    def load_model(self):
        """Load ONNX model for Supertrend Pullback (V4.8)."""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            self.model = onnxruntime.InferenceSession(self.model_path)
            logging.info(f"✅ Loaded Supertrend Pullback V4.8 ONNX model: {os.path.basename(self.model_path)}")
        except Exception as e:
            logging.exception(f"❌ Error loading Supertrend Pullback V4.8 model: {e}")
            raise

    def load_scaler(self):
        """Load scaler for Supertrend Pullback (V4.8)."""
        try:
            if not os.path.exists(self.scaler_path):
                raise FileNotFoundError(f"Scaler file not found: {self.scaler_path}")

            with open(self.scaler_path, 'rb') as f:
                scalers = pickle.load(f)

            base_symbol = self.contract_symbol.split('.')[0][:2] 
            
            if base_symbol in scalers:
                self.scaler = scalers[base_symbol]
                logging.info(f"✅ Loaded '{base_symbol}' scaler for Supertrend V4.8")
            else:
                available = list(scalers.keys())
                raise ValueError(
                    f"'{base_symbol}' scaler not found. "
                    f"Available: {available}"
                )
        except Exception as e:
            logging.exception(f"❌ Error loading Supertrend V4.8 scaler: {e}")
            raise

    # --- (predict and should_enter_trade are correct for pure AI mode) ---
    def predict(self, df: pd.DataFrame) -> Tuple[int, float]:
        """
        Generate prediction using V4.8 Transformer model (ONNX).
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
            logging.exception(f"❌ Prediction error (Supertrend V4.8): {e}")
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
        Determine if V4.8 "Pure AI Mode" entry conditions are met.
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