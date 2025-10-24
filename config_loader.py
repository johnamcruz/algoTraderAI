"""
Configuration File Loader - YAML Only

Loads YAML config files for trading bot.
Command-line arguments override config file values.
"""

import os
from typing import Dict, Any


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
    Command-line args take precedence.
    
    Args:
        config: Dictionary from config file
        args: argparse Namespace object
        
    Returns:
        Merged configuration dictionary
    """
    # Start with config file values
    merged = config.copy()
    
    # Override with command-line arguments (if provided)
    args_dict = vars(args)
    for key, value in args_dict.items():
        # Only override if argument was explicitly provided
        # (not just using argparse default)
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
    required = ['account', 'contract', 'username', 'apikey', 'strategy', 'model', 'scaler']
    
    missing = [key for key in required if key not in config or config[key] is None]
    
    if missing:
        raise ValueError(
            f"Missing required configuration values: {', '.join(missing)}\n"
            f"Provide them in config file or via command-line arguments."
        )