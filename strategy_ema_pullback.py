#!/usr/bin/env python3
"""
EMA Pullback Strategy Implementation (V1.0 - Transformer)
This strategy is based on the final, outcome-weighted V1.0 configuration (30 Features)
Hardened version to prevent feature dropping during calculation and ensures only 30
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


class EmaPullbackStrategy(BaseStrategy):
    """
    EMA Pullback V1.0 (Transformer) trading strategy.
    Hardened feature engineering for production stability.
    """

    def __init__(self, model_path: str, scaler_path: str, contract_symbol: str):
        super().__init__(model_path, scaler_path, contract_symbol)
        logging.info("Initialized EmaPullbackStrategy (V1.0 - 30 Features - Hardened)")
        self.mtf_history: Dict[str, pd.DataFrame] = {}


    def get_feature_columns(self) -> List[str]:
        """
        Returns the 30 feature columns for the final V1.0 EMA Pullback Transformer.
        """
        # ⭐️ V1.0 FEATURE COLS: 30 FEATURES ⭐️
        return [
            # TIER 1: CORE TREND & MTF (11)
            'st_direction', 'st_direction_slow', 'price_vs_st', 'price_vs_st_slow',
            'adx', 'price_vs_ema40', 'price_vs_ema200',
            'mtf_alignment_score', 'macro_trend_slope', 'macro_trend_direction_60m', 'ema40_direction',

            # TIER 2: PULLBACK & REGIME QUALITY (5)
            'tradeable_trend', 'pullback_orderly', 'fresh_structure',
            'ema_separation', 'volatility_regime',

            # TIER 3: MOMENTUM & VOLUME (4)
            'price_roc_slope', 'volume_thrust',
            'stoch_rsi', 'mfi',

            # TIER 4: CANDLE & RISK (5)
            'atr', 'body_size', 'normalized_bb_width', 'hour_sin', 'hour_cos',

            # TIER 5: CONTEXT & RISK (3)
            'day_of_week_encoded', 'setup_quality_tier', 'setup_quality_score',

            # ⭐️ NEW EMA FEATURES (ADDED) (2) ⭐️
            'ema9_vs_ema20',
            'price_vs_ema20',
        ]

    def get_sequence_length(self) -> int:
        return 120

    # =========================================================
    # V1.0 FEATURE ENGINEERING (30 FEATURES) - HARDENED
    # =========================================================
    def _calculate_setup_quality_tiers(self, df: pd.DataFrame, labels_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate multi-tier quality scores WITHOUT filtering.
        Returns quality tier (0=worst, 4=best) for each bar.
        
        NOTE: This is a helper for add_features. We use .values[1:] to align
        with the labeling script's logic, but in live trading, we only care about the last value.
        The [1:labels_length+1] slicing is kept for consistency with the training data calc.
        For live trading, a simpler version might be needed if labels_length is not available,
        but this will work for backtesting/prediction runs.
        
        Let's simplify for live prediction. We only need the *most recent* values.
        """
        
        # Simplified version for live prediction (operates on the whole dataframe)
        
        # Extract relevant arrays
        adx = df['adx'].values
        mtf_alignment = df['mtf_alignment_score'].values
        volume_thrust = df['volume_thrust'].values
        fresh_structure = df['fresh_structure'].values
        pullback_orderly = df['pullback_orderly'].values
        volatility_regime = df['volatility_regime'].values

        # Tier scoring (each condition = 1 point)
        quality_score = np.zeros(len(df), dtype=np.float32)

        # Tier 1: Trend Strength (0-2 points)
        quality_score += (adx > 20).astype(float)
        quality_score += (adx > 30).astype(float)  # Extra point for strong trend

        # Tier 2: Multi-timeframe (0-2 points)
        quality_score += (np.abs(mtf_alignment) >= 2).astype(float)  # 2 of 3 aligned
        quality_score += (np.abs(mtf_alignment) == 3).astype(float)  # All aligned

        # Tier 3: Volume & Structure (0-2 points)
        quality_score += (volume_thrust > 1.2).astype(float)
        quality_score += (fresh_structure == 1).astype(float)

        # Tier 4: Regime Quality (0-2 points)
        quality_score += (pullback_orderly == 1).astype(float)
        quality_score += ((volatility_regime > 0.8) & (volatility_regime < 1.3)).astype(float)

        # Convert to tiers (0-4)
        tiers = np.zeros(len(df), dtype=np.int8)
        tiers[quality_score >= 2] = 1
        tiers[quality_score >= 4] = 2
        tiers[quality_score >= 6] = 3
        tiers[quality_score == 8] = 4

        return tiers, quality_score
    
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
        Calculate V1.0 features with robust error handling and cleanup.
        This version ensures OHLCV and intermediate columns are NOT dropped,
        allowing the base class to function correctly.
        """
        logging.debug(f"Adding V1.0 EMA Pullback features. Input shape: {df.shape}")

        # ⭐️ CRITICAL: Operate on a copy
        df = df.copy()

        # --- 1. Core Indicators (3-min TF) ---
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14).replace(0, 1e-6)

        # ⭐️ ADDED EMA(9) and EMA(20) ⭐️
        df['ema9'] = ta.ema(df['close'], length=9)
        df['ema20'] = ta.ema(df['close'], length=20)

        df['ema15'] = ta.ema(df['close'], length=15); df['ema40'] = ta.ema(df['close'], length=40); df['ema200'] = ta.ema(df['close'], length=200)

        st_df_fast = ta.supertrend(df['high'], df['low'], df['close'], length=10, multiplier=3)
        df['st_val'] = st_df_fast['SUPERT_10_3']; df['st_direction'] = st_df_fast['SUPERTd_10_3']
        st_df_slow = ta.supertrend(df['high'], df['low'], df['close'], length=20, multiplier=4)
        df['st_val_slow'] = st_df_slow['SUPERT_20_4']; df['st_direction_slow'] = st_df_slow['SUPERTd_20_4']

        # --- 2. Normalized Price/Momentum Features (3-min TF) ---
        df['price_vs_st'] = (df['close'] - df['st_val']) / df['atr'];
        df['price_vs_st_slow'] = (df['close'] - df['st_val_slow']) / df['atr'];

        # ⭐️ NEW EMA NORMALIZED FEATURES ⭐️
        df['price_vs_ema20'] = (df['close'] - df['ema20']) / df['atr']
        df['ema9_vs_ema20'] = (df['ema9'] - df['ema20']) / df['atr']

        df['price_vs_ema40'] = (df['close'] - df['ema40']) / df['atr'];
        # --- REMOVED --- 'ema15_vs_ema40' is redundant
        df['price_vs_ema200'] = (df['close'] - df['ema200']) / df['atr']

        adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
        df['adx'] = adx_df['ADX_14'].fillna(0)

        # --- REMOVED --- 'adx_acceleration_5'

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

        # Volume features
        df['_calc_volume_ma20'] = df['volume'].rolling(20).mean()
        df['volume_thrust'] = df['volume'] / (df['_calc_volume_ma20'] + 1e-6)
        # --- REMOVED --- 'volume_declining'

        # Time-Based Contextual Features (Cyclical)
        df['_calc_hour'] = df.index.hour
        hours_in_day = 24
        df['hour_sin'] = np.sin(2 * np.pi * df['_calc_hour'] / hours_in_day)
        df['hour_cos'] = np.cos(2 * np.pi * df['_calc_hour'] / hours_in_day)
        df['_calc_day_of_week'] = df.index.dayofweek
        df['day_of_week_encoded'] = df['_calc_day_of_week'] / 6.0

        # --- 3. V1.0 Pullback & Regime Quality Features ---

        # Regime/Trend
        df['_calc_st_consistency_20'] = df['st_direction'].rolling(20).sum() / 20
        df['tradeable_trend'] = (df['_calc_st_consistency_20'].abs() >= 0.7).astype(int)
        
        df['ema_separation'] = (df['ema15'] - df['ema40']).abs() / df['atr']

        df['_calc_atr_ma20'] = df['atr'].rolling(20).mean()
        df['volatility_regime'] = df['atr'] / (df['_calc_atr_ma20'] + 1e-6)
        # --- REMOVED --- 'volatility_spike'

        # Pullback Quality
        df['pullback_orderly'] = (df['body_size'].rolling(3).std() < df['body_size'].rolling(20).mean() * 0.5).astype(int)

        # Structure
        df['_calc_swing_high'] = ((df['high'] > df['high'].shift(1)) &
                             (df['high'] > df['high'].shift(-1))).astype(int)
        df['_calc_swing_low'] = ((df['low'] < df['low'].shift(1)) &
                            (df['low']< df['low'].shift(-1))).astype(int)
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
            df_15m_direction = df_15m_features[['ema40_direction']].copy()
            df_15m_direction.rename(columns={'ema40_direction': '_calc_ema40_direction_15m'}, inplace=True)
            df = pd.merge_asof(df, df_15m_direction, left_index=True, right_index=True, direction='forward')
            df['ema40_direction'] = df['_calc_ema40_direction_15m'].fillna(0.0)
            # df.drop(columns=['_calc_ema40_direction_15m'], inplace=True, errors='ignore') # Keep for cleanup later

        _15m_slope_col = f'_calc_ema40_slope_15m'
        if _15m_slope_col in df_15m_features.columns:
            df_15m_slope = df_15m_features[[_15m_slope_col]].copy()
            df = pd.merge_asof(df, df_15m_slope, left_index=True, right_index=True, direction='forward')
            df['macro_trend_slope'] = df[_15m_slope_col] / df['atr']
            # df.drop(columns=[_15m_slope_col], inplace=True, errors='ignore') # Keep for cleanup later
        else:
            df['macro_trend_slope'] = 0.0

        # 4b. 60M Strategic Trend
        df_60m_features = self._calculate_mtf_slope_hardened(df, '60Min')

        if 'macro_trend_direction_60m' in df_60m_features.columns:
            df_60m_direction = df_60m_features[['macro_trend_direction_60m']].copy()
            df_60m_direction.rename(columns={'macro_trend_direction_60m': '_calc_macro_direction_60m'}, inplace=True)
            df = pd.merge_asof(df, df_60m_direction, left_index=True, right_index=True, direction='forward')
            df['macro_trend_direction_60m'] = df['_calc_macro_direction_60m'].fillna(0.0)
            # df.drop(columns=['_calc_macro_direction_60m'], inplace=True, errors='ignore') # Keep for cleanup later

        # MTF Alignment Score
        df['mtf_alignment_score'] = (
            df['st_direction'] +
            df['ema40_direction'] +
            df['macro_trend_direction_60m']
        )

        # --- 5. Composite Quality Scores (V1.0) ---
        # We must fillna *before* calling the quality tier function
        df.fillna(method='ffill', limit=200, inplace=True)
        df.fillna(0, inplace=True) # Fill zeros *before* quality calc

        tiers, scores = self._calculate_setup_quality_tiers(df, len(df))
        df['setup_quality_tier'] = tiers
        df['setup_quality_score'] = scores # This is the primary score from the helper
        
        # --- 6. Final Robust Cleanup ---
        # Robust NaN/Inf Handling
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Final fillna pass
        df.fillna(method='ffill', limit=200, inplace=True)
        df.fillna(method='bfill', limit=200, inplace=True)
        df.fillna(0, inplace=True)

        # --- 7. Final Feature Verification ---
        # (Ensures columns exist, but does NOT drop OHLCV)
        final_feature_cols = self.get_feature_columns()
        missing_cols = []
        for col in final_feature_cols:
            if col not in df.columns:
                df[col] = 0.0 # Add the missing column back as zeros
                missing_cols.append(col)

        if missing_cols:
            logging.warning(f"⚠️ Added missing V1.0 features as zeros (likely due to insufficient history): {', '.join(missing_cols)}")

        logging.debug(f"V1.0 EMA Pullback features added. Shape after features: {df.shape}")

        return df

    # =========================================================
    # LIVE EXECUTION FUNCTIONS (ONNX)
    # =========================================================

    def load_model(self):
        """Load ONNX model for EMA Pullback (V1.0)."""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            self.model = onnxruntime.InferenceSession(self.model_path)
            logging.info(f"✅ Loaded EMA Pullback V1.0 ONNX model: {os.path.basename(self.model_path)}")
        except Exception as e:
            logging.exception(f"❌ Error loading EMA Pullback V1.0 model: {e}")
            raise

    def load_scaler(self):
        """Load scaler for EMA Pullback (V1.0)."""
        try:
            if not os.path.exists(self.scaler_path):
                raise FileNotFoundError(f"Scaler file not found: {self.scaler_path}")

            with open(self.scaler_path, 'rb') as f:
                scalers = pickle.load(f)

            base_symbol = self.contract_symbol.split('.')[0][:2] 
            
            if base_symbol in scalers:
                self.scaler = scalers[base_symbol]
                logging.info(f"✅ Loaded '{base_symbol}' scaler for V1.0")
            else:
                available = list(scalers.keys())
                raise ValueError(
                    f"'{base_symbol}' scaler not found. "
                    f"Available: {available}"
                )
        except Exception as e:
            logging.exception(f"❌ Error loading V1.0 scaler: {e}")
            raise

    # --- (predict and should_enter_trade are correct for pure AI mode) ---
    def predict(self, df: pd.DataFrame) -> Tuple[int, float]:
        """
        Generate prediction using V1.0 Transformer model (ONNX).
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

            # ⭐️ Ensure correct feature count (30)
            if features.shape[1] != len(self.get_feature_columns()):
                 logging.error(f"❌ Feature dimension mismatch! Model expects {len(self.get_feature_columns())}, but got {features.shape[1]}")
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
            logging.exception(f"❌ Prediction error (EMA Pullback V1.0): {e}")
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
        Determine if V1.0 "Pure AI Mode" entry conditions are met.
        ⭐️ SELL SIGNALS ARE DISABLED ⭐️
        """
        
        # 1. Confidence Filter (Primary Filter)
        if confidence < entry_conf:
            return False, None
                    
        # Check prediction
        if prediction == 1:  # Buy signal
            return True, 'LONG'
        #elif prediction == 2:  # Sell signal
        #    return True, 'SHORT'

        # Prediction is 0 (Hold)
        return False, None

    @staticmethod
    def _softmax(x):
        """Compute softmax values for array x."""
        if x is None or len(x) == 0:
            return np.array([])
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()