#!/usr/bin/env python3
"""
Multi-Timeframe Directional Strategy Implementation (V4.0) - FIXED FOR LIVE

CRITICAL FIX: Uses ROLLING WINDOWS (not resampling) to match training

DATA REQUIREMENTS:
- Minimum: 250 bars (~12.5 hours / 1.5 trading days)
  - Needed for slow_window=200 + sequence=80
- Recommended: 300+ bars (~15 hours / 2 days) for stable features
- Full quality: 400+ bars for all calculations to stabilize

WINDOW SIZES (MUST MATCH TRAINING):
- Fast: 20 bars (60 minutes)
- Medium: 60 bars (180 minutes / 3 hours)
- Slow: 200 bars (600 minutes / 10 hours)
"""

import pandas as pd
import numpy as np
import onnxruntime
import pickle
import os
import logging
from typing import Dict, List, Tuple, Optional
from strategy_base import BaseStrategy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class MultiTFDirectionalStrategy(BaseStrategy):
    """Multi-Timeframe Directional V4.0 - FIXED for Live Trading"""

    def __init__(self, model_path: str, scaler_path: str, contract_symbol: str):
        """Initialize Multi-TF Directional Strategy V4.0"""
        super().__init__(model_path, scaler_path, contract_symbol)
        
        self.bar_count = 0
        self.last_quality = None
        
        logging.info("=" * 70)
        logging.info("🚀 Multi-Timeframe Directional Strategy V4.0 - FIXED")
        logging.info("=" * 70)
        logging.info("📊 Features: 23 multi-timeframe indicators")
        logging.info("🎯 Sequence: 80 bars")
        logging.info("🔧 Feature Calculation: ROLLING WINDOWS (matches training)")
        logging.info("📈 Data Requirements:")
        logging.info("   - Minimum: 250 bars (~12.5 hours / 1.5 days)")
        logging.info("   - Recommended: 300+ bars for stable features")
        logging.info("   - Full quality: 400+ bars")
        logging.info("⚡ Expected Performance @ 0.55 threshold:")
        logging.info("   - BUY Precision: 78.7%")
        logging.info("   - Signal Rate: 0.5% (~0.7 trades/day)")
        logging.info("   - Win Rate: ~75%+ (4R targets)")
        logging.info("=" * 70)

    def get_feature_columns(self) -> List[str]:
        """Returns the 23 multi-timeframe feature columns for V4.0"""
        return [
            'fast_momentum', 'fast_trend_strength', 'fast_volatility_regime',
            'fast_volume_regime', 'fast_price_position',
            'med_momentum', 'med_trend_strength', 'med_volatility_regime',
            'med_volume_regime', 'med_price_position',
            'slow_momentum', 'slow_trend_strength', 'slow_volatility_regime',
            'slow_volume_regime', 'slow_price_position',
            'tf_momentum_alignment', 'tf_trend_consistency', 'tf_volatility_divergence',
            'orderflow_proxy', 'liquidity_regime',
            'hour_sin', 'hour_cos', 'regime_stability',
        ]

    def get_sequence_length(self) -> int:
        """V4.0 uses 80 bars"""
        return 80
    
    def get_quality_level(self, bars: int) -> str:
        """Return data quality level based on bar count"""
        if bars < 250:
            return "INSUFFICIENT"
        elif bars < 300:
            return "WARMING_UP"
        elif bars < 400:
            return "GOOD"
        else:
            return "EXCELLENT"
    
    def _calculate_regime_stability(self, alignment_series: pd.Series) -> np.ndarray:
        """
        Calculate regime stability looking BACKWARD only.
        NO LOOKAHEAD BIAS - matches training code exactly.
        
        For each bar, counts consecutive bars with same alignment value.
        Only looks at current and past bars.
        
        Args:
            alignment_series: Series of tf_momentum_alignment values
            
        Returns:
            Array of stability values (0-1 range)
        """
        stability = np.zeros(len(alignment_series))
        
        if len(alignment_series) == 0:
            return stability
        
        # Start with first bar
        current_value = alignment_series.iloc[0]
        count = 1
        stability[0] = 1
        
        # Iterate forward through time (no lookahead!)
        for i in range(1, len(alignment_series)):
            if alignment_series.iloc[i] == current_value:
                # Same regime continues
                count += 1
            else:
                # Regime changed, reset counter
                current_value = alignment_series.iloc[i]
                count = 1
            
            stability[i] = count
        
        # Normalize to 0-1 range (cap at 20 bars max)
        return np.minimum(stability / 20.0, 1.0)

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate V4.0 Multi-Timeframe features using ROLLING WINDOWS
        
        CRITICAL: This MUST match the training calculation method!
        Uses rolling windows on 3-minute data (NOT resampling to 15min/1hr)
        
        Window sizes:
        - fast_window = 20 bars
        - med_window = 60 bars
        - slow_window = 200 bars
        """
        self.bar_count += 1
        current_bars = len(df)
        quality = self.get_quality_level(current_bars)
        
        # Log quality changes
        if quality != self.last_quality:
            if quality == "INSUFFICIENT":
                logging.warning(f"⚠️ INSUFFICIENT DATA: {current_bars} bars (need 250+)")
            elif quality == "WARMING_UP":
                logging.info(f"🔥 WARMING UP: {current_bars} bars - predictions starting")
            elif quality == "GOOD":
                logging.info(f"✅ GOOD QUALITY: {current_bars} bars - features stable")
            elif quality == "EXCELLENT":
                logging.info(f"🎯 EXCELLENT: {current_bars} bars - full quality achieved")
            self.last_quality = quality
        
        df = df.copy()
        
        # Window sizes MUST match training
        fast_window = 20
        med_window = 60
        slow_window = 200
        
        # =====================================================================
        # TRUE RANGE CALCULATION (base for all trend strength)
        # =====================================================================
        
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        # =====================================================================
        # FAST TIMEFRAME (20 bars = 60 minutes)
        # =====================================================================
        
        # 1. Fast Momentum
        df['fast_momentum'] = df['close'].pct_change(fast_window)
        
        # 2. Fast Trend Strength (ADX-like)
        plus_di = 100 * (plus_dm.rolling(fast_window).mean() / true_range.rolling(fast_window).mean())
        minus_di = 100 * (minus_dm.rolling(fast_window).mean() / true_range.rolling(fast_window).mean())
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        df['fast_trend_strength'] = dx.rolling(fast_window).mean() / 100.0
        
        # 3. Fast Volatility Regime
        atr_fast = true_range.rolling(fast_window).mean()
        df['fast_volatility_regime'] = atr_fast / df['close']
        
        # 4. Fast Volume Regime
        df['fast_volume_regime'] = (
            df['volume'] / df['volume'].rolling(fast_window).mean()
        ) - 1.0
        
        # 5. Fast Price Position
        rolling_high = df['high'].rolling(fast_window).max()
        rolling_low = df['low'].rolling(fast_window).min()
        df['fast_price_position'] = (
            (df['close'] - rolling_low) / (rolling_high - rolling_low + 1e-10)
        )
        
        # =====================================================================
        # MEDIUM TIMEFRAME (60 bars = 3 hours)
        # =====================================================================
        
        # 6. Med Momentum
        df['med_momentum'] = df['close'].pct_change(med_window)
        
        # 7. Med Trend Strength
        plus_di_med = 100 * (plus_dm.rolling(med_window).mean() / true_range.rolling(med_window).mean())
        minus_di_med = 100 * (minus_dm.rolling(med_window).mean() / true_range.rolling(med_window).mean())
        dx_med = 100 * np.abs(plus_di_med - minus_di_med) / (plus_di_med + minus_di_med + 1e-10)
        df['med_trend_strength'] = dx_med.rolling(med_window).mean() / 100.0
        
        # 8. Med Volatility Regime
        atr_med = true_range.rolling(med_window).mean()
        df['med_volatility_regime'] = atr_med / df['close']
        
        # 9. Med Volume Regime
        df['med_volume_regime'] = (
            df['volume'] / df['volume'].rolling(med_window).mean()
        ) - 1.0
        
        # 10. Med Price Position
        rolling_high_med = df['high'].rolling(med_window).max()
        rolling_low_med = df['low'].rolling(med_window).min()
        df['med_price_position'] = (
            (df['close'] - rolling_low_med) / (rolling_high_med - rolling_low_med + 1e-10)
        )
        
        # =====================================================================
        # SLOW TIMEFRAME (200 bars = 10 hours)
        # =====================================================================
        
        # 11. Slow Momentum
        df['slow_momentum'] = df['close'].pct_change(slow_window)
        
        # 12. Slow Trend Strength
        plus_di_slow = 100 * (plus_dm.rolling(slow_window).mean() / true_range.rolling(slow_window).mean())
        minus_di_slow = 100 * (minus_dm.rolling(slow_window).mean() / true_range.rolling(slow_window).mean())
        dx_slow = 100 * np.abs(plus_di_slow - minus_di_slow) / (plus_di_slow + minus_di_slow + 1e-10)
        df['slow_trend_strength'] = dx_slow.rolling(slow_window).mean() / 100.0
        
        # 13. Slow Volatility Regime
        atr_slow = true_range.rolling(slow_window).mean()
        df['slow_volatility_regime'] = atr_slow / df['close']
        
        # 14. Slow Volume Regime
        df['slow_volume_regime'] = (
            df['volume'] / df['volume'].rolling(slow_window).mean()
        ) - 1.0
        
        # 15. Slow Price Position
        rolling_high_slow = df['high'].rolling(slow_window).max()
        rolling_low_slow = df['low'].rolling(slow_window).min()
        df['slow_price_position'] = (
            (df['close'] - rolling_low_slow) / (rolling_high_slow - rolling_low_slow + 1e-10)
        )
        
        # =====================================================================
        # CROSS-TIMEFRAME FEATURES
        # =====================================================================
        
        # 16. TF Momentum Alignment
        df['tf_momentum_alignment'] = (
            np.sign(df['fast_momentum']) + 
            np.sign(df['med_momentum']) + 
            np.sign(df['slow_momentum'])
        ) / 3.0
        
        # 17. TF Trend Consistency
        trend_mean = (df['fast_trend_strength'] + df['med_trend_strength'] + df['slow_trend_strength']) / 3.0
        trend_std = np.sqrt(
            ((df['fast_trend_strength'] - trend_mean) ** 2 +
             (df['med_trend_strength'] - trend_mean) ** 2 +
             (df['slow_trend_strength'] - trend_mean) ** 2) / 3.0
        )
        df['tf_trend_consistency'] = 1.0 / (1.0 + trend_std)
        
        # 18. TF Volatility Divergence
        df['tf_volatility_divergence'] = (
            df['fast_volatility_regime'] - df['slow_volatility_regime']
        )
        
        # =====================================================================
        # MICROSTRUCTURE FEATURES
        # =====================================================================
        
        # 19. Order Flow Proxy
        df['orderflow_proxy'] = (
            (df['close'] - df['low']) - (df['high'] - df['close'])
        ) / (df['high'] - df['low'] + 1e-10)
        
        # 20. Liquidity Regime
        df['liquidity_regime'] = (
            df['volume'].rolling(fast_window).mean() * 
            df['fast_volatility_regime']
        )
        df['liquidity_regime'] = df['liquidity_regime'] / df['liquidity_regime'].rolling(100).mean()
        
        # =====================================================================
        # TEMPORAL FEATURES
        # =====================================================================
        
        # 21-22. Hour encoding
        if isinstance(df.index, pd.DatetimeIndex):
            hour = df.index.hour + df.index.minute / 60.0
        else:
            hour = pd.Series(12.0, index=df.index)
        
        df['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
        df['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
        
        # 23. Regime Stability (matches training exactly)
        # Count how many bars current regime has persisted
        df['regime_stability'] = self._calculate_regime_stability(df['tf_momentum_alignment'])
        
        # =====================================================================
        # CLEANUP
        # =====================================================================
        
        df = df.replace([np.inf, -np.inf], np.nan)
        
        feature_cols = self.get_feature_columns()
        for col in feature_cols:
            if col in df.columns:
                df[col] = df[col].fillna(method='ffill').fillna(0)
        
        return df

    def load_model(self):
        """Load ONNX model"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            self.model = onnxruntime.InferenceSession(self.model_path)
            logging.info(f"✅ Loaded Multi-TF V4.0 model: {os.path.basename(self.model_path)}")
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
            
            # Extract base symbol (ES or NQ)
            base_symbol = self.contract_symbol.split('.')[0][:2]
            
            if base_symbol in scalers:
                self.scaler = scalers[base_symbol]
                logging.info(f"✅ Loaded '{base_symbol}' scaler for Multi-TF V4.0")
            else:
                available = list(scalers.keys())
                raise ValueError(f"'{base_symbol}' scaler not found. Available: {available}")
        except Exception as e:
            logging.exception(f"❌ Error loading scaler: {e}")
            raise

    def predict(self, df: pd.DataFrame) -> Tuple[int, float]:
        """
        Generate prediction from Multi-TF V4.0 model
        
        Returns:
            prediction: 0=HOLD, 1=BUY, 2=SELL
            confidence: Model confidence (0-1)
        """
        try:
            seq_len = self.get_sequence_length()
            current_bars = len(df)
            
            # Need minimum bars for slow_window + sequence
            if df.empty or current_bars < 250:
                if self.bar_count % 50 == 0:  # Log every 50 bars
                    logging.info(f"⏳ Accumulating data: {current_bars}/250 bars")
                return 0, 0.0

            features = self.preprocess_features(df)
            if len(features) < seq_len:
                return 0, 0.0

            X = features[-seq_len:].reshape(1, seq_len, -1).astype(np.float32)
            
            if np.isnan(X).any() or np.isinf(X).any():
                logging.error(f"❌ Invalid values detected in input")
                return 0, 0.0

            # Run inference
            outputs = self.model.run(None, {'input_sequence': X})
            
            # Get logits and confidence
            logits_sequence = outputs[0]
            last_logits = logits_sequence[0, -1, :]
            
            # Get confidence from auxiliary head (output 2)
            confidence_sequence = outputs[2]
            confidence_pred = float(confidence_sequence[0, -1])
            
            probs = self._softmax(last_logits)
            prediction = int(np.argmax(probs))
            
            # Use auxiliary confidence (not max prob)
            confidence = confidence_pred
            
            pred_name = ['HOLD', 'BUY', 'SELL'][prediction]
            quality = self.get_quality_level(current_bars)
            
            # Log signals and periodic status
            if prediction != 0:
                logging.info(f"🎯 [{quality}] {pred_name} (Conf: {confidence:.3f}) | Bars: {current_bars}")
            elif self.bar_count % 100 == 0:
                logging.debug(f"📊 [{quality}] {pred_name} (Conf: {confidence:.3f}) | Bars: {current_bars}")

            return prediction, confidence

        except Exception as e:
            logging.exception(f"❌ Prediction error: {e}")
            return 0, 0.0

    def should_enter_trade(
        self,
        prediction: int,
        confidence: float,
        bar: Dict,
        entry_conf: float = 0.55,
        adx_thresh: float = 0.0,
    ) -> Tuple[bool, Optional[str]]:
        """
        Determine if entry conditions are met for Multi-TF V4.0
        
        VALIDATED THRESHOLD: 0.55
        - Expected precision: 78.7% (at full quality)
        - Expected signal rate: 0.5% (~0.7 trades/day)
        - Expected win rate: ~75%+ with 4R targets
        
        Args:
            prediction: Model prediction (0=HOLD, 1=BUY, 2=SELL)
            confidence: Model confidence (0-1)
            bar: Current bar data (not used, kept for compatibility)
            entry_conf: Confidence threshold (default 0.55)
            adx_thresh: Not used (kept for compatibility)
        
        Returns:
            (should_enter, direction): (True, 'LONG') or (False, None)
        """
        if confidence < entry_conf:
            return False, None
        
        quality = self.get_quality_level(self.bar_count)
        
        # Only trade BUY signals
        if prediction == 1:
            logging.info(f"✅ [{quality}] BUY SIGNAL | Conf: {confidence:.3f}")
            if quality == "EXCELLENT":
                logging.info(f"   Expected: 78.7% precision @ 0.55 threshold")
                logging.info(f"   Win Rate: ~75%+ with 4R targets")
            elif quality == "GOOD":
                logging.info(f"   Expected: ~75% precision (good quality)")
            else:
                logging.info(f"   Expected: ~70% precision (warming up)")
            return True, 'LONG'
        
        elif prediction == 2:
            # Log SELL signals but don't trade them
            if confidence > 0.60:
                logging.info(f"ℹ️ [{quality}] SELL signal (conf: {confidence:.3f}) - risk-off indicator")
            return False, None
        
        return False, None

    @staticmethod
    def _softmax(x):
        """Compute softmax values for array x"""
        if x is None or len(x) == 0:
            return np.array([])
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()