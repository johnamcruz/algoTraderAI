#!/usr/bin/env python3
"""
Multi-Horizon Directional Strategy Implementation (V3.4)
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


class MultiHorizonDirectionalStrategy(BaseStrategy):
    """Multi-Horizon Directional V3.4 trading strategy."""

    def __init__(self, model_path: str, scaler_path: str, contract_symbol: str):
        """Initialize MultiHorizonDirectionalStrategy V3.4"""
        super().__init__(model_path, scaler_path, contract_symbol)
        logging.info("Initialized Multi-Horizon Directional Strategy (V3.4)")

    def get_feature_columns(self) -> List[str]:
        """Returns the 12 feature columns for V3.4"""
        return [
            'rsi', 'macd_hist', 'price_roc', 'close_norm',
            'ema_spread', 'adx', 'price_vs_ema',
            'atr_norm', 'atr_regime',
            'volume_ratio',
            'hour_sin', 'hour_cos',
        ]

    def get_sequence_length(self) -> int:
        """V3.4 uses 60 bars"""
        return 60

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate V3.4 Multi-Horizon features (12 features)"""
        if len(df) < 200:
            logging.warning(f"⚠️ Insufficient data. Need 200+, have {len(df)}")
        
        df = df.copy()
        
        # 1. ATR (volatility base)
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14).replace(0, 1e-6)
        
        # 2. RSI
        df['rsi'] = ta.rsi(df['close'], length=14).fillna(50)
        
        # 3. MACD Histogram
        macd_df = ta.macd(df['close'])
        if macd_df is not None and 'MACDh_12_26_9' in macd_df.columns:
            df['macd_hist'] = macd_df['MACDh_12_26_9'].fillna(0)
        else:
            df['macd_hist'] = 0.0
        
        # 4. Price Rate of Change
        df['price_roc'] = df['close'].pct_change(5).fillna(0)
        
        # 5. Close Normalization
        rolling_min = df['close'].rolling(200, min_periods=20).min()
        rolling_max = df['close'].rolling(200, min_periods=20).max()
        df['close_norm'] = (df['close'] - rolling_min) / (rolling_max - rolling_min + 1e-6)
        df['close_norm'] = df['close_norm'].fillna(0.5)
        
        # 6-7. EMA Spread and Price vs EMA
        ema_fast = ta.ema(df['close'], length=20)
        ema_slow = ta.ema(df['close'], length=50)
        df['ema_spread'] = (ema_fast - ema_slow) / df['atr']
        df['price_vs_ema'] = (df['close'] - ema_fast) / df['atr']
        
        # 8. ADX
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
        if adx_df is not None and 'ADX_14' in adx_df.columns:
            df['adx'] = adx_df['ADX_14'].fillna(0)
        else:
            df['adx'] = 0.0
        
        # 9-10. Volatility Regime
        df['atr_norm'] = df['atr'] / df['close']
        atr_ma = df['atr'].rolling(20).mean()
        df['atr_regime'] = df['atr'] / (atr_ma + 1e-6)
        
        # 11. Volume Ratio
        vol_ma = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / (vol_ma + 1e-6)
        
        # 12-13. Time Encoding
        df['hour'] = df.index.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Cleanup
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        df.fillna(0, inplace=True)
        df.drop(columns=['hour'], inplace=True, errors='ignore')
        
        return df

    def load_model(self):
        """Load ONNX model"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            self.model = onnxruntime.InferenceSession(self.model_path)
            logging.info(f"✅ Loaded model: {os.path.basename(self.model_path)}")
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
                logging.info(f"✅ Loaded '{base_symbol}' scaler")
            else:
                available = list(scalers.keys())
                raise ValueError(f"'{base_symbol}' scaler not found. Available: {available}")
        except Exception as e:
            logging.exception(f"❌ Error loading scaler: {e}")
            raise

    def predict(self, df: pd.DataFrame) -> Tuple[int, float]:
        """Generate prediction"""
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
            
            if np.isnan(X).any() or np.isinf(X).any():
                logging.error(f"❌ Invalid values detected")
                return 0, 0.0

            input_name = self.model.get_inputs()[0].name
            output_name = self.model.get_outputs()[0].name
            logits_sequence = self.model.run([output_name], {input_name: X})[0]
            
            last_logits = logits_sequence[0, -1, :]
            probs = self._softmax(last_logits)
            prediction = int(np.argmax(probs))
            confidence = float(probs[prediction])
            
            pred_name = ['HOLD', 'BUY', 'SELL'][prediction]
            logging.info(f"🎯 {pred_name} ({confidence:.3f})")

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
        if confidence < entry_conf:
            logging.debug(f"⏭️ Rejected: conf {confidence:.3f} < {entry_conf}")
            return False, None
                
        if prediction == 1:
            logging.info(f"✅ BUY SIGNAL ACCEPTED | Conf: {confidence:.3f}")
            return True, 'LONG'
        elif prediction == 2:
            logging.info(f"✅ SELL SIGNAL ACCEPTED | Conf: {confidence:.3f}")
            return True, 'SHORT'
        
        return False, None

    @staticmethod
    def _softmax(x):
        """Compute softmax values for array x"""
        if x is None or len(x) == 0:
            return np.array([])
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()