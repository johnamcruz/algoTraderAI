#!/usr/bin/env python3
"""
Squeeze V3 Strategy Implementation

This strategy uses BB/KC squeeze detection with momentum analysis.
"""

import pandas as pd
import pandas_ta as ta
import numpy as np
import onnxruntime
import pickle
import os
from typing import Dict, List, Tuple, Optional
from strategy_base import BaseStrategy


class SqueezeV3Strategy(BaseStrategy):
    """
    Squeeze V3 trading strategy based on Bollinger Band / Keltner Channel compression.
    """
    
    def get_feature_columns(self) -> List[str]:
        """Returns feature columns for Squeeze V3 strategy."""
        return [
            'compression_level', 'squeeze_duration', 'bb_expanding',
            'atr_expanding', 'price_in_range', 'rsi',
            'compressed_momentum', 'vol_surge', 'body_strength',
        ]
    
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Squeeze V3 features."""
        # Ensure numeric types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        # ATR
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        
        # Bollinger Bands
        bbands = ta.bbands(df['close'], length=20, std=2)
        bb_upper_col = next((col for col in bbands.columns if col.startswith('BBU_')), 'BBU_20_2.0_2.0')
        bb_lower_col = next((col for col in bbands.columns if col.startswith('BBL_')), 'BBL_20_2.0_2.0')
        bb_mid_col = next((col for col in bbands.columns if col.startswith('BBM_')), 'BBM_20_2.0_2.0')
        df['bb_upper'] = bbands[bb_upper_col]
        df['bb_lower'] = bbands[bb_lower_col]
        df['bb_mid'] = bbands[bb_mid_col]
        
        # Keltner Channels
        kc = ta.kc(df['high'], df['low'], df['close'], length=20, scalar=1.5)
        kc_upper_col = next((col for col in kc.columns if col.startswith('KCU')), 'KCUe_20_1.5')
        kc_lower_col = next((col for col in kc.columns if col.startswith('KCL')), 'KCLe_20_1.5')
        df['kc_upper'] = kc[kc_upper_col]
        df['kc_lower'] = kc[kc_lower_col]
        
        # RSI
        df['rsi'] = ta.rsi(df['close'], length=14)
        
        # ADX
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
        df['adx'] = adx_df['ADX_14']

        # === SQUEEZE FEATURES ===
        # Squeeze detection
        df['squeeze_on'] = (df['bb_lower'] > df['kc_lower']) & (df['bb_upper'] < df['kc_upper'])
        
        # Compression level
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_width_pct'] = df['bb_width'] / df['close']
        df['compression_level'] = df['bb_width_pct'].rolling(20).apply(
            lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-10), raw=False
        )
        
        # Squeeze duration
        df['squeeze_duration'] = 0
        for i in range(1, len(df)):
            if df.iloc[i]['squeeze_on']:
                df.iloc[i, df.columns.get_loc('squeeze_duration')] = \
                    df.iloc[i-1]['squeeze_duration'] + 1 if df.iloc[i-1]['squeeze_on'] else 1
        
        # BB expanding
        df['bb_expanding'] = (df['bb_width'].diff() > 0).astype(float)
        
        # ATR expanding
        df['atr_expanding'] = (df['atr'].diff() > 0).astype(float)
        
        # Price position in range
        df['price_in_range'] = (df['close'] - df['bb_lower']) / (df['bb_width'] + 1e-10)
        
        # Compressed momentum
        df['momentum'] = df['close'] - df['close'].shift(10)
        df['compressed_momentum'] = df['momentum'] / (df['atr'] + 1e-10)
        
        # Volume surge
        df['vol_ma'] = df['volume'].rolling(20).mean()
        df['vol_surge'] = df['volume'] / (df['vol_ma'] + 1e-10)
        
        # Body strength
        df['body_size'] = abs(df['close'] - df['open'])
        df['candle_range'] = df['high'] - df['low']
        df['body_strength'] = df['body_size'] / (df['candle_range'] + 1e-10)

        # Clean up
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        df.fillna(0, inplace=True)
        
        return df
    
    def load_model(self):
        """Load ONNX model for Squeeze V3."""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            self.model = onnxruntime.InferenceSession(self.model_path)
            print(f"✅ Loaded Squeeze V3 model: {os.path.basename(self.model_path)}")
        except Exception as e:
            print(f"❌ Error loading Squeeze V3 model: {e}")
            raise
    
    def load_scaler(self):
        """Load scaler for Squeeze V3."""
        try:
            if not os.path.exists(self.scaler_path):
                raise FileNotFoundError(f"Scaler file not found: {self.scaler_path}")
            
            with open(self.scaler_path, 'rb') as f:
                scalers = pickle.load(f)
            
            if self.contract_symbol in scalers:
                self.scaler = scalers[self.contract_symbol]
                print(f"✅ Loaded '{self.contract_symbol}' scaler for Squeeze V3")
            else:
                available = list(scalers.keys())
                raise ValueError(
                    f"'{self.contract_symbol}' scaler not found. "
                    f"Available: {available}"
                )
        except Exception as e:
            print(f"❌ Error loading Squeeze V3 scaler: {e}")
            raise
    
    def predict(self, df: pd.DataFrame) -> Tuple[int, float]:
        """Generate prediction using Squeeze V3 model."""
        try:
            # Preprocess features
            features = self.preprocess_features(df)
            
            # Prepare input for ONNX (needs sequence dimension)
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
            print(f"❌ Prediction error (Squeeze V3): {e}")
            return 0, 0.0
    
    def should_enter_trade(
        self, 
        prediction: int, 
        confidence: float,
        bar: Dict,
        entry_conf: float,
        adx_thresh: float
    ) -> Tuple[bool, Optional[str]]:
        """Determine if Squeeze V3 entry conditions are met."""
        
        # Check confidence threshold
        if confidence < entry_conf:
            return False, None
        
        # Check ADX threshold (if ADX available in bar)
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