#!/usr/bin/env python3
"""
Supertrend Pullback Strategy Implementation (V4.9 - 30 Features)
FINAL VERSION with optimized logging levels
"""

import pandas as pd
import pandas_ta as ta
import numpy as np
import onnxruntime
import pickle
import os
import logging
from typing import Dict, List, Tuple, Optional
from strategy_base import BaseStrategy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SupertrendPullback3Strategy(BaseStrategy):
    """
    Supertrend Pullback V4.9 (30 Features) trading strategy.
    FIXED for real-time execution with optimized logging.
    """

    def __init__(self, model_path: str, scaler_path: str, contract_symbol: str):
        """Initialize SupertrendPullbackStrategy V4.9"""
        super().__init__(model_path, scaler_path, contract_symbol)
        logging.info("Initialized SupertrendPullbackStrategy (V4.9 - 30 Features)")
        
        # Cache for MTF data to avoid recalculation
        self._last_15m_timestamp = None
        self._last_60m_timestamp = None
        self._cached_15m_slope = 0.0
        self._cached_60m_slope = 0.0
        self._cached_60m_direction = 0
        self._cached_15m_direction = 0

    def get_feature_columns(self) -> List[str]:
        """Returns the 30 feature columns for V4.9"""
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

            # NEW EMA FEATURES (2)
            'ema9_vs_ema20',
            'price_vs_ema20',
        ]

    def get_sequence_length(self) -> int:
        """V4.9 uses 60 bars"""
        return 60

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate V4.9 Supertrend features - optimized for real-time execution.
        FIXED with proper lookahead prevention.
        """
        logging.debug(f"🔧 Adding V4.9 features. Input shape: {df.shape}")
        
        if len(df) < 200:
            logging.warning(f"⚠️ Insufficient data for feature calculation. Need 200+, have {len(df)}")
        
        df = df.copy()
        
        # --- 1. CORE INDICATORS ---
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14).replace(0, 1e-6)
        
        df['ema9'] = ta.ema(df['close'], length=9)
        df['ema15'] = ta.ema(df['close'], length=15)
        df['ema20'] = ta.ema(df['close'], length=20)
        df['ema40'] = ta.ema(df['close'], length=40)
        df['ema200'] = ta.ema(df['close'], length=200)
        
        # SuperTrend indicators
        st_df_fast = ta.supertrend(df['high'], df['low'], df['close'], length=10, multiplier=3)
        df['st_val'] = st_df_fast['SUPERT_10_3']
        df['st_direction'] = st_df_fast['SUPERTd_10_3']
        
        st_df_slow = ta.supertrend(df['high'], df['low'], df['close'], length=20, multiplier=4)
        df['st_val_slow'] = st_df_slow['SUPERT_20_4']
        df['st_direction_slow'] = st_df_slow['SUPERTd_20_4']
        
        # --- 2. NORMALIZED PRICE FEATURES ---
        df['price_vs_st'] = (df['close'] - df['st_val']) / df['atr']
        df['price_vs_st_slow'] = (df['close'] - df['st_val_slow']) / df['atr']
        df['ema9_vs_ema20'] = (df['ema9'] - df['ema20']) / df['atr']
        df['price_vs_ema20'] = (df['close'] - df['ema20']) / df['atr']
        df['price_vs_ema40'] = (df['close'] - df['ema40']) / df['atr']
        df['price_vs_ema200'] = (df['close'] - df['ema200']) / df['atr']
        
        # --- 3. MOMENTUM & TREND STRENGTH ---
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
        df['adx'] = adx_df['ADX_14'].fillna(0) if adx_df is not None and 'ADX_14' in adx_df.columns else 0
        
        df['price_vel_20'] = df['close'].diff(20) / df['atr']
        df['price_roc_slope'] = df['price_vel_20'].diff(10)
        
        # --- 4. CANDLE FEATURES ---
        df['body_size'] = abs(df['close'] - df['open']) / df['atr']
        df['wick_ratio'] = (df['high'] - df['low']) / df['atr']
        
        # Bollinger Bands
        bbands = ta.bbands(df['close'], length=20, std=2)
        if bbands is not None and 'BBU_20_2.0' in bbands.columns:
            df['normalized_bb_width'] = (bbands['BBU_20_2.0'] - bbands['BBL_20_2.0']) / (df['atr'] * 2 + 1e-6)
        else:
            df['normalized_bb_width'] = 0.0
        df['normalized_bb_width'] = df['normalized_bb_width'].fillna(0.0)
        
        # --- 5. VOLUME FEATURES ---
        vol_std = df['volume'].rolling(10).std().replace(0, 1e-6)
        df['volume_velocity'] = df['volume'].diff(1) / vol_std
        
        df['volume_ma20'] = df['volume'].rolling(20).mean()
        df['volume_thrust'] = df['volume'] / (df['volume_ma20'] + 1e-6)
        
        mfi_df = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=14)
        df['mfi'] = mfi_df.fillna(50) if mfi_df is not None else 50
        
        # StochRSI
        stochrsi_df = ta.stochrsi(df['close'], length=14)
        if stochrsi_df is not None and 'STOCHRSIk_14_14_3_3' in stochrsi_df.columns:
            df['stoch_rsi'] = stochrsi_df['STOCHRSIk_14_14_3_3'].fillna(0)
        else:
            df['stoch_rsi'] = 0.0
        
        # --- 6. TIME FEATURES ---
        df['hour'] = df.index.hour
        hours_in_day = 24
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / hours_in_day)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / hours_in_day)
        
        df['day_of_week'] = df.index.dayofweek
        df['day_of_week_encoded'] = df['day_of_week'] / 6.0
        
        # --- 7. MARKET REGIME FEATURES ---
        df['st_consistency_20'] = df['st_direction'].rolling(20).sum() / 20
        df['tradeable_trend'] = (df['st_consistency_20'].abs() >= 0.7).astype(int)
        
        df['ema_separation'] = (df['ema15'] - df['ema40']).abs() / df['atr']
        
        df['atr_ma20'] = df['atr'].rolling(20).mean()
        df['volatility_regime'] = df['atr'] / (df['atr_ma20'] + 1e-6)
        
        df['pullback_orderly'] = (
            df['body_size'].rolling(3).std() < df['body_size'].rolling(20).mean() * 0.5
        ).astype(int)
        
        # --- 8. STRUCTURE FEATURES (FIXED FOR REAL-TIME) ---
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['rejection_wick_long'] = (df['lower_wick'] / (df['atr'] + 1e-6) > 0.5).astype(int)
        df['rejection_wick_short'] = (df['upper_wick'] / (df['atr'] + 1e-6) > 0.5).astype(int)
        
        # 🔧 FIXED: Remove lookahead bias from swing detection
        df['swing_high'] = (
            (df['high'] > df['high'].shift(1)) & 
            (df['high'] > df['high'].shift(2))  # FIXED: was shift(-1)
        ).astype(int)
        df['swing_low'] = (
            (df['low'] < df['low'].shift(1)) & 
            (df['low'] < df['low'].shift(2))  # FIXED: was shift(-1)
        ).astype(int)
        
        swing_marker = (df['swing_high'] | df['swing_low']).astype(bool)
        df['bars_since_swing'] = df.groupby(swing_marker.cumsum()).cumcount()
        df['fresh_structure'] = (df['bars_since_swing'] <= 10).astype(int)
        
        df['recent_whipsaw'] = (df['st_direction'].diff().abs().rolling(10).sum() > 2).astype(int)
        
        # --- 9. MTF FEATURES ---
        df = self._add_mtf_features_optimized(df)
        
        # --- 10. SETUP QUALITY SCORES ---
        df = self._calculate_setup_quality(df)
        
        # --- 11. CLEANUP ---
        drop_cols = ['hour', 'day_of_week', 'atr_ma20', 'volume_ma20', 
                     'swing_high', 'swing_low', 'st_consistency_20',
                     'rejection_wick_long', 'rejection_wick_short',
                     'recent_whipsaw', 'price_vel_20', 'volume_velocity',
                     'upper_wick', 'lower_wick', 'bars_since_swing', 'wick_ratio']
        
        feature_set = set(self.get_feature_columns())
        drop_cols = [c for c in drop_cols if c in df.columns and c not in feature_set]
        if drop_cols:
            df.drop(columns=drop_cols, inplace=True, errors='ignore')
        
        # --- 12. HANDLE NaN/Inf ---
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.ffill(inplace=True)
        df.fillna(0, inplace=True)
        
        # Ensure all feature columns exist
        missing_features = []
        for col in self.get_feature_columns():
            if col not in df.columns:
                missing_features.append(col)
                df[col] = 0.0
        
        if missing_features:
            logging.error(f"❌ Missing features: {missing_features}")
        
        # 🔍 LOG FINAL FEATURE CHECK (only key values)
        last_bar = df.iloc[-1]
        logging.debug(f"✅ Features shape: {df.shape}")
        logging.info(f"📊 ST:{last_bar['st_direction']:.0f} ADX:{last_bar['adx']:.1f} MTF:{last_bar['mtf_alignment_score']:.0f}")
        
        return df

    def _add_mtf_features_optimized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add MTF features with caching"""
        
        if len(df) < 40:
            logging.warning("⚠️ Not enough data for MTF, using defaults")
            df['ema40_slope'] = 0.0
            df['ema40_direction'] = 0
            df['ema40_slope_60m'] = 0.0
            df['macro_trend_direction_60m'] = 0
            df['macro_trend_slope'] = 0.0
            df['mtf_alignment_score'] = df['st_direction']
            return df
        
        # Fix duplicate timestamps
        if not df.index.is_unique:
            dup_count = df.index.duplicated().sum()
            logging.warning(f"⚠️ Duplicate timestamps: {dup_count}, removing...")
            df = df[~df.index.duplicated(keep='last')]
            
        current_15m_bar = df.index[-1].floor('15Min')
        current_60m_bar = df.index[-1].floor('60Min')
        
        # --- 15-MINUTE MTF ---
        if self._last_15m_timestamp != current_15m_bar or self._cached_15m_slope == 0.0:
            try:
                df_15m = df['close'].resample('15Min', label='left', closed='left').ohlc().dropna()
                
                if not df_15m.empty and 'close' in df_15m.columns and len(df_15m) >= 43:
                    ema_15m = ta.ema(df_15m['close'], length=40)
                    
                    if ema_15m is not None and len(ema_15m) > 3:
                        ema_15m_shifted = ema_15m.shift(1).dropna()
                        
                        if len(ema_15m_shifted) > 0:
                            slope = ema_15m_shifted.diff(3).iloc[-1]
                            self._cached_15m_slope = slope if not pd.isna(slope) else 0.0
                            self._cached_15m_direction = 1 if self._cached_15m_slope > 0 else -1
                            self._last_15m_timestamp = current_15m_bar
                            logging.debug(f"🔄 15m MTF updated: dir={self._cached_15m_direction}")
                        else:
                            logging.debug("⚠️ 15m EMA shifted empty")
                    else:
                        logging.debug("⚠️ 15m EMA calculation failed")
                else:
                    logging.debug(f"⚠️ Not enough 15m bars: {len(df_15m)}/43")
            except Exception as e:
                logging.error(f"❌ 15m MTF error: {e}")
        
        # --- 60-MINUTE MTF ---
        if self._last_60m_timestamp != current_60m_bar or self._cached_60m_slope == 0.0:
            try:
                df_60m = df['close'].resample('60Min', label='left', closed='left').ohlc().dropna()
                
                if not df_60m.empty and 'close' in df_60m.columns and len(df_60m) >= 43:
                    ema_60m = ta.ema(df_60m['close'], length=40)
                    
                    if ema_60m is not None and len(ema_60m) > 3:
                        ema_60m_shifted = ema_60m.shift(1).dropna()
                        
                        if len(ema_60m_shifted) > 0:
                            slope = ema_60m_shifted.diff(3).iloc[-1]
                            self._cached_60m_slope = slope if not pd.isna(slope) else 0.0
                            self._cached_60m_direction = 1 if self._cached_60m_slope > 0 else -1
                            self._last_60m_timestamp = current_60m_bar
                            logging.debug(f"🔄 60m MTF updated: dir={self._cached_60m_direction}")
                        else:
                            logging.debug("⚠️ 60m EMA shifted empty")
                    else:
                        logging.debug("⚠️ 60m EMA calculation failed")
                else:
                    logging.debug(f"⚠️ Not enough 60m bars: {len(df_60m)}/43")
            except Exception as e:
                logging.error(f"❌ 60m MTF error: {e}")
        
        # Broadcast cached values
        df['ema40_slope'] = self._cached_15m_slope
        df['ema40_direction'] = self._cached_15m_direction
        df['ema40_slope_60m'] = self._cached_60m_slope
        df['macro_trend_direction_60m'] = self._cached_60m_direction
        
        atr_safe = df['atr'].replace(0, 1e-6)
        df['macro_trend_slope'] = df['ema40_slope'] / atr_safe
        
        df['mtf_alignment_score'] = (
            df['st_direction'] + 
            df['ema40_direction'] + 
            df['macro_trend_direction_60m']
        )                
        return df

    def _calculate_setup_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate setup quality tier and score"""
        quality_score = np.zeros(len(df), dtype=np.float32)
        
        adx = df['adx'].values
        st_direction = df['st_direction'].values
        mtf_alignment = df['mtf_alignment_score'].values
        pullback_orderly = df['pullback_orderly'].values
        fresh_structure = df['fresh_structure'].values
        recent_whipsaw = df['recent_whipsaw'].values
        volatility_regime = df['volatility_regime'].values
        volume_thrust = df['volume_thrust'].values
        stoch_rsi = df['stoch_rsi'].values
        ema_separation = df['ema_separation'].values
        
        # DIMENSION 1: TREND CONFLUENCE
        quality_score += (np.abs(mtf_alignment) >= 2).astype(float)
        quality_score += (np.abs(mtf_alignment) == 3).astype(float)
        quality_score += (adx > 25).astype(float)
        
        # DIMENSION 2: PULLBACK & STRUCTURE
        quality_score += (pullback_orderly == 1).astype(float)
        quality_score += (fresh_structure == 1).astype(float)
        quality_score += (recent_whipsaw == 0).astype(float)
        
        # DIMENSION 3: REGIME & VOLUME
        stable_vol = (volatility_regime > 0.7) & (volatility_regime < 1.3)
        quality_score += stable_vol.astype(float)
        
        light_volume = (volume_thrust < 0.95)
        quality_score += light_volume.astype(float)
        
        momentum_extreme = (
            ((st_direction > 0) & (stoch_rsi < 30)) |
            ((st_direction < 0) & (stoch_rsi > 70))
        )
        quality_score += momentum_extreme.astype(float)
        
        # DIMENSION 4: HIGH-CONVICTION BONUS
        adx_super_strong = (adx > 30) & (ema_separation > 0.5)
        quality_score += adx_super_strong.astype(float)
        
        # Convert to tiers
        tiers = np.zeros(len(df), dtype=np.int8)
        tiers[quality_score >= 3] = 1
        tiers[quality_score >= 5] = 2
        tiers[quality_score >= 7] = 3
        tiers[quality_score >= 8] = 4
        tiers[quality_score >= 9] = 5
        
        df['setup_quality_tier'] = tiers
        df['setup_quality_score'] = quality_score
        
        return df

    # =========================================================
    # LIVE EXECUTION FUNCTIONS
    # =========================================================

    def load_model(self):
        """Load ONNX model"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            self.model = onnxruntime.InferenceSession(self.model_path)
            logging.info(f"✅ Loaded V4.9 model: {os.path.basename(self.model_path)}")
        except Exception as e:
            logging.exception(f"❌ Error loading model: {e}")
            raise

    def load_scaler(self):
        """Load scaler"""
        try:
            if not os.path.exists(self.scaler_path):
                raise FileNotFoundError(f"Scaler file not found: {self.scaler_path}")

            with open(self.scaler_path, 'rb') as f:
                scalers = pickle.load(f)

            base_symbol = self.contract_symbol.split('.')[0][:2]
            
            if base_symbol in scalers:
                self.scaler = scalers[base_symbol]
                logging.info(f"✅ Loaded '{base_symbol}' scaler for V4.9")
            else:
                available = list(scalers.keys())
                raise ValueError(
                    f"'{base_symbol}' scaler not found. Available: {available}"
                )
        except Exception as e:
            logging.exception(f"❌ Error loading scaler: {e}")
            raise

    def predict(self, df: pd.DataFrame) -> Tuple[int, float]:
        """Generate prediction using V4.9 Transformer model"""
        try:
            seq_len = self.get_sequence_length()
            
            if df.empty or len(df) < seq_len:
                logging.error(f"❌ Not enough data. Need {seq_len}, have {len(df)}")
                return 0, 0.0

            features = self.preprocess_features(df)

            if len(features) < seq_len:
                logging.error(f"❌ Data mismatch after scaling. Need {seq_len}, have {len(features)}")
                return 0, 0.0

            X = features[-seq_len:].reshape(1, seq_len, -1).astype(np.float32)
            
            # Check for NaN/Inf
            if np.isnan(X).any():
                nan_count = np.isnan(X).sum()
                logging.error(f"❌ NaN detected: {nan_count} values")
                return 0, 0.0
            
            if np.isinf(X).any():
                inf_count = np.isinf(X).sum()
                logging.error(f"❌ Inf detected: {inf_count} values")
                return 0, 0.0

            # Run inference
            input_name = self.model.get_inputs()[0].name
            output_name = self.model.get_outputs()[0].name
            logits_sequence = self.model.run([output_name], {input_name: X})[0]

            last_logits = logits_sequence[0, -1, :]
            probs = self._softmax(last_logits)
            prediction = int(np.argmax(probs))
            confidence = float(probs[prediction])
            
            # Compact prediction logging
            pred_name = ['HOLD', 'BUY', 'SELL'][prediction]
            logging.info(f"🎯 {pred_name} ({confidence:.3f}) | H:{probs[0]:.2f} B:{probs[1]:.2f} S:{probs[2]:.2f}")

            return prediction, confidence

        except Exception as e:
            logging.exception(f"❌ Prediction error: {e}")
            return 0, 0.0

    def should_enter_trade(
        self,
        prediction: int,
        confidence: float,
        bar: Dict,
        entry_conf: float,
        adx_thresh: float,
    ) -> Tuple[bool, Optional[str]]:
        """Determine if entry conditions are met"""
        
        # Just confidence check (as intended)
        if confidence < entry_conf:
            logging.debug(f"⏭️ Rejected: conf {confidence:.3f} < {entry_conf}")
            return False, None
                
        if prediction == 1:  # BUY
            logging.info(f"✅ BUY SIGNAL ACCEPTED | Conf: {confidence:.3f}")
            return True, 'LONG'
        
        return False, None

    @staticmethod
    def _softmax(x):
        """Compute softmax values for array x"""
        if x is None or len(x) == 0:
            return np.array([])
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()