#!/usr/bin/env python3
"""
Base Strategy Class for AI Trading Bot

This module defines the abstract base class that all trading strategies must implement.
Each strategy handles its own features, AI model, and prediction logic.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.
    
    Each strategy must implement:
    - Feature calculation
    - AI prediction logic
    - Entry signal generation
    """
    
    def __init__(self, model_path: str, contract_symbol: str):
        """
        Initialize base strategy.

        Args:
            model_path: Path to ONNX model file
            contract_symbol: Trading symbol (ES, NQ, YM, RTY)
        """
        self.model_path = model_path
        self.contract_symbol = contract_symbol
        self.model = None
        self._bar_count: int = 0

    def set_contract_symbol(self, symbol):
        """Method to set the contract symbol after initialization."""
        self.contract_symbol = symbol        
        
    @abstractmethod
    def get_feature_columns(self) -> List[str]:
        """
        Returns the list of feature column names used by this strategy.
        
        Returns:
            List of feature column names
        """
        pass
    
    @abstractmethod
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate and add all features required by this strategy.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added feature columns
        """
        pass
    
    @abstractmethod
    def load_model(self):
        """
        Load the ONNX model for this strategy.
        Should set self.model to the loaded model.
        """
        pass
    

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> Tuple[int, float]:
        """
        Generate prediction from the AI model.
        
        Args:
            df: DataFrame with calculated features
            
        Returns:
            Tuple of (prediction, confidence)
            - prediction: 0=Hold, 1=Buy, 2=Sell
            - confidence: Probability/confidence score (0-1)
        """
        pass
    
    @abstractmethod
    def should_enter_trade(
        self, 
        prediction: int, 
        confidence: float,
        bar: Dict,
        entry_conf: float,
        adx_thresh: float
    ) -> Tuple[bool, Optional[str]]:
        """
        Determine if entry conditions are met.
        
        Args:
            prediction: Model prediction (0=Hold, 1=Buy, 2=Sell)
            confidence: Model confidence score
            bar: Current bar data with features
            entry_conf: Minimum confidence threshold
            adx_thresh: Minimum ADX threshold
            
        Returns:
            Tuple of (should_enter, direction)
            - should_enter: Boolean indicating if trade should be entered
            - direction: 'LONG' or 'SHORT' if should_enter is True, else None
        """
        pass
    
    def _on_new_bar(self, df: pd.DataFrame, bar_idx: int) -> None:
        """
        Called once per bar for stateful incremental processing.
        No-op by default. Override in strategies that maintain running state
        (e.g. zone detectors, pivot trackers) that must see every bar in order.
        """
        pass

    def _run_warmup(self, df: pd.DataFrame) -> None:
        """
        Feed each of the n-1 historical bars through _on_new_bar so that
        stateful strategies build correct state from prefilled history before
        going live. Safe to call every add_features() — exits immediately after
        the first call (when _bar_count > 0).
        """
        n = len(df)
        for warmup_i in range(n - 1):
            self._on_new_bar(df.iloc[:warmup_i + 1], warmup_i)
            self._bar_count += 1

    def on_trade_exit(self, reason: str):
        """Called by the bot when a trade exits. Override to react to exit events."""
        pass

    def is_trading_allowed(self, timestamp: pd.Timestamp) -> bool:
        """
        Return False to block entry at this bar's timestamp.
        Default: always allowed. Override in strategies with session restrictions.
        """
        return True

    def get_stop_target_pts(
        self,
        df: pd.DataFrame,
        direction: str,
        entry_price: float,
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Return strategy-derived (stop_pts, target_pts) for the upcoming trade.
        Return (None, None) to use the bot's global --stop_pts / --target_pts instead.
        Strategies that know their natural stop (e.g. zone-based) should override this.
        """
        return None, None

    def get_sequence_length(self) -> int:
        """
        Returns the ONNX model input window length (bars per inference).
        Override per strategy to match the model's expected sequence dimension.
        """
        return 60

    def get_warmup_length(self) -> int:
        """
        Returns the number of historical bars the bot should prefill before
        going live. Must be >= get_sequence_length(). Override per strategy
        when indicators need a longer lookback than the model sequence.
        """
        return max(200, self.get_sequence_length())
    
    def validate_features(self, df: pd.DataFrame) -> bool:
        """
        Validate that all required features are present in the dataframe.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if all features present, False otherwise
        """
        required_features = self.get_feature_columns()
        missing = [col for col in required_features if col not in df.columns]
        
        if missing:
            print(f"❌ Missing features: {missing}")
            return False
        return True
    
    def preprocess_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Common preprocessing for features before model input.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Numpy array ready for model input
        """
        feature_cols = self.get_feature_columns()
        features = df[feature_cols].values
        
        # Handle NaN values
        features = np.nan_to_num(features, nan=0.0)
        
        return features