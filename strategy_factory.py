#!/usr/bin/env python3
"""
Strategy Factory

Creates strategy instances based on strategy name.
"""

from strategy_base import BaseStrategy
from strategy_squeeze import SqueezeV3Strategy
from strategy_pivot_reversal_5min import PivotReversal5minStrategy
from strategy_pivot_reversal_3min import PivotReversal3minStrategy
from strategy_vwap_3min import VWAP3minStrategy
from strategy_trend_pullback import TrendPullbackStrategy

class StrategyFactory:
    """
    Factory class for creating strategy instances.
    """
    
    # Registry of available strategies
    STRATEGIES = {
        'squeeze_v3': SqueezeV3Strategy,
        '5min_pivot_reversal': PivotReversal5minStrategy,
        '3min_pivot_reversal': PivotReversal3minStrategy,
        'vwap': VWAP3minStrategy,
        'trend_pullback': TrendPullbackStrategy,
    }
    
    @classmethod
    def create_strategy(
        cls,
        strategy_name: str,
        model_path: str,
        scaler_path: str,
        contract_symbol: str,
        **kwargs
    ) -> BaseStrategy:
        """
        Create a strategy instance.
        
        Args:
            strategy_name: Name of the strategy ('squeeze_v3', 'pivot_reversal', etc.)
            model_path: Path to ONNX model file
            scaler_path: Path to pickled scaler file
            contract_symbol: Trading symbol (ES, NQ, YM, RTY)
            **kwargs: Additional strategy-specific parameters
            
        Returns:
            Strategy instance
            
        Raises:
            ValueError: If strategy name is not recognized
        """
        strategy_name = strategy_name.lower()
        
        if strategy_name not in cls.STRATEGIES:
            available = ', '.join(cls.STRATEGIES.keys())
            raise ValueError(
                f"Unknown strategy: '{strategy_name}'. "
                f"Available strategies: {available}"
            )
        
        strategy_class = cls.STRATEGIES[strategy_name]
        return strategy_class(model_path, scaler_path, contract_symbol, **kwargs)
    
    @classmethod
    def list_strategies(cls) -> list:
        """Return list of available strategy names."""
        return list(cls.STRATEGIES.keys())
    
    @classmethod
    def register_strategy(cls, name: str, strategy_class: type):
        """
        Register a new strategy class.
        
        Args:
            name: Strategy name
            strategy_class: Strategy class (must inherit from BaseStrategy)
        """
        if not issubclass(strategy_class, BaseStrategy):
            raise TypeError(
                f"Strategy class must inherit from BaseStrategy, "
                f"got {strategy_class.__name__}"
            )
        
        cls.STRATEGIES[name.lower()] = strategy_class
        print(f"✅ Registered strategy: '{name}'")