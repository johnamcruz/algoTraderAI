"""
Configuration File Loader - YAML Only

Loads YAML config files for trading bot.
Command-line arguments override config file values.
"""

import os
from typing import Dict, Any

DEFAULTS = {
    'size':                 1,
    'timeframe':            5,
    'entry_conf':           0.9,
    'adx_thresh':           0,
    'tick_size':            0.01,
    'profit_target':        6000,
    'max_loss':             3000,
    'market_hub':           'https://rtc.topstepx.com/hubs/market',
    'base_url':             'https://api.topstepx.com/api',
    'strategy':             'cisd-ote',
    'model':                'models/cisd_ote_hybrid_v5_1.onnx',
    'scaler':               'models/scaler_supertrend_pullback_v3.10.pkl',
    'high_conf_multiplier': 1.0,
    'max_contracts':        15,
    'pivot_lookback':       8,
    'min_vty_regime':       0.75,
    'enable_trailing_stop': False,
    'breakeven_on_1r':      False,
    'min_stop_pts':         1.0,
    'min_stop_atr':         0.5,
}


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file (.yaml or .yml)
        
    Returns:
        Dictionary of configuration values
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If file format is unsupported
        ImportError: If PyYAML is not installed
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    file_ext = os.path.splitext(config_path)[1].lower()
    
    if file_ext not in ['.yaml', '.yml']:
        raise ValueError(f"Only YAML config files are supported. Got: {file_ext}")
    
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required for config files.\n"
            "Install it with: pip install pyyaml"
        )
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config if config else {}


def merge_config_with_args(config: Dict[str, Any], args: Any) -> Dict[str, Any]:
    """
    Merge config file with command-line arguments.
    Priority: CLI args > YAML config > hardcoded defaults.
    All argparse defaults must be None so they don't silently override YAML.
    """
    # Start with hardcoded defaults
    merged = DEFAULTS.copy()

    # Layer YAML config on top
    merged.update(config)

    # Layer explicit CLI args on top (only non-None values were user-provided)
    for key, value in vars(args).items():
        if value is not None:
            merged[key] = value

    return merged


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate required configuration values.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValueError: If required values are missing
    """
    required = ['account', 'contract', 'username', 'apikey', 'strategy', 'model', 'scaler', 'market_hub', 'base_url']
    
    missing = [key for key in required if key not in config or config[key] is None]
    
    if missing:
        raise ValueError(
            f"Missing required configuration values: {', '.join(missing)}\n"
            f"Provide them in config file or via command-line arguments."
        )