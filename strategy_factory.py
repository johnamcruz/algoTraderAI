#!/usr/bin/env python3
"""
Strategy Factory

Creates strategy instances based on strategy name.
"""

from strategy_base import BaseStrategy
from strategy_cisd_ote import CISDOTEStrategy

def _load_v7():
    from strategy_cisd_ote_v7 import CISDOTEStrategyV7
    return CISDOTEStrategyV7

class StrategyFactory:
    """
    Factory class for creating strategy instances.
    """

    # Registry of available strategies
    STRATEGIES = {
        'cisd-ote':  CISDOTEStrategy,
        'cisd-ote7': _load_v7,   # lazy: futures_foundation only required when v7 is selected
    }
    
    @classmethod
    def create_strategy(
        cls,
        strategy_name: str,
        model_path: str,
        contract_symbol: str,
        **kwargs
    ) -> BaseStrategy:
        """
        Create a strategy instance.

        Args:
            strategy_name: Name of the strategy
            model_path: Path to ONNX model file
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
        if callable(strategy_class) and not isinstance(strategy_class, type):
            strategy_class = strategy_class()  # resolve lazy loader
        return strategy_class(model_path, contract_symbol, **kwargs)
    
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