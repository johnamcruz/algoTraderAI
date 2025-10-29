#!/usr/bin/env python3
"""
Real-Time Trading Bot with Pluggable Strategies

This version supports multiple AI strategies through a clean strategy pattern.
Each strategy handles its own features, predictions, and entry logic.

Requirements:
    pip install onnxruntime pandas pandas-ta signalrcore requests numpy scikit-learn python-dateutil
"""

import asyncio
import argparse
import logging

# Import local modules
from bot_utils import setup_logging, authenticate, MARKET_HUB, BASE_URL
from config_loader import load_config, merge_config_with_args, validate_config
from strategy_factory import StrategyFactory
#from strategy_base import BaseStrategy
from trading_bot import RealTimeBot
import warnings

warnings.filterwarnings('ignore')

# =========================================================
# MAIN
# =========================================================
def main():
    setup_logging(level=logging.INFO, log_file="bot_log.log")

    parser = argparse.ArgumentParser(
        description='Real-Time AI Futures Trading Bot with Pluggable Strategies',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=f"""
Available Strategies: {', '.join(StrategyFactory.list_strategies())}

Example Usage (Squeeze V3):
  python %(prog)s --account ACC123 --contract CON.F.US.RTY.Z25 --size 1 \\
                  --username USER --apikey KEY --timeframe 5 \\
                  --strategy squeeze_v3 \\
                  --model models/squeeze_v3_model.onnx \\
                  --scaler models/squeeze_v3_scalers.pkl \\
                  --entry_conf 0.55 --adx_thresh 25 --stop_atr 2.0 --target_atr 3.0

Example Usage (Pivot Reversal):
  python %(prog)s --account ACC123 --contract CON.F.US.ES.Z25 --size 1 \\
                  --username USER --apikey KEY --timeframe 5 \\
                  --strategy pivot_reversal \\
                  --model models/pivot_reversal_model.onnx \\
                  --scaler models/pivot_reversal_scalers.pkl \\
                  --entry_conf 0.70 --adx_thresh 20 --stop_atr 1.0 --target_atr 2.0
"""
    )

    # Config file option
    parser.add_argument('--config', type=str, 
                        help='Path to YAML config file. Command-line args override config values.')
    
    # Account & Contract
    parser.add_argument('--account', type=str,
                        help='TopstepX account ID')
    parser.add_argument('--contract', type=str,
                        help='Full contract ID (e.g., CON.F.US.ENQ.Z25)')
    parser.add_argument('--size', type=int, default=1,
                        help='Trade size (number of contracts)')
    parser.add_argument('--username', type=str,
                        help='TopstepX username')
    parser.add_argument('--apikey', type=str,
                        help='TopstepX API key')
    parser.add_argument('--timeframe', type=int, choices=[1, 3, 5], default=3,
                        help='Bar timeframe in minutes (default: 3)')    
    
    #RealTime Market URL and Base URL
    parser.add_argument('--market_hub', type=str, default=MARKET_HUB,
                        help='Market Hub URL')
    parser.add_argument('--base_url', type=str, default=BASE_URL,
                        help='ProjectX Base URL')
    
    # Strategy Selection
    parser.add_argument('--strategy', type=str, default="3min_pivot_reversal",
                        choices=StrategyFactory.list_strategies(),
                        help='Strategy to use')
    parser.add_argument('--model', type=str, default="models/model_3min_pivot_reversal_v2_final.onnx",
                        help='Path to the ONNX model file (.onnx)')
    parser.add_argument('--scaler', type=str, default="models/scalers_3min_pivot_reversal_v2_final.pkl",
                        help='Path to the pickled scaler file (.pkl)')
    
    # Trading Parameters
    parser.add_argument('--entry_conf', type=float, default=0.60,
                        help='Min AI confidence to enter (default: 0.80)')
    parser.add_argument('--adx_thresh', type=int, default=20,
                        help='Min ADX value to enter (default: 20)')
    parser.add_argument('--stop_atr', type=float, default=1.5,
                        help='Stop loss multiplier (x ATR) (default: 1.0)')
    parser.add_argument('--target_atr', type=float, default=2.0,
                        help='Profit target multiplier (x ATR) (default: 2.0)')
    parser.add_argument('--enable_trailing_stop', action='store_true',
                        help='Enable trailing stop vs stop order')
    
    # Strategy-specific parameters
    parser.add_argument('--pivot_lookback', type=int, default=8,
                        help='Pivot lookback period (for pivot_reversal strategy)')
    
    args = parser.parse_args()

    # Load configuration
    if args.config:
        # Config file provided - load and merge with args
        try:
            config = load_config(args.config)
            config = merge_config_with_args(config, args)
            validate_config(config)
            logging.info(f"Loaded config from: {args.config}")
        except (FileNotFoundError, ValueError, ImportError) as e:
            logging.exception(f"Configuration error: {e}")
            return
    else:
        # No config file - use args directly
        config = vars(args)        
        config.pop('config', None)
        
        # Validate required fields
        try:
            validate_config(config)
        except ValueError as e:            
            parser.print_help()
            return
    
    # log config data for debugging
    logging.info(config)

    # Authenticate
    jwt_token = authenticate(config["base_url"], config["username"], config["apikey"])
    if not jwt_token:
        return
    
    logging.info(f"🎫 Token received. Creating strategy...")
    
    try:
        
        # Create strategy
        strategy_kwargs = {}
        if args.strategy == '3min_pivot_reversal' or args.strategy == '5min_pivot_reversal':
            strategy_kwargs['pivot_lookback'] = config["pivot_lookback"]
        
        strategy = StrategyFactory.create_strategy(
            strategy_name=config["strategy"],
            model_path=config["model"],
            scaler_path=config["scaler"],
            contract_symbol=None,
            **strategy_kwargs
        )
        
        logging.info(f"✅ Strategy '{args.strategy}' created successfully!")
        logging.info(f"Starting bot...")
        
        # Create and run bot
        bot = RealTimeBot(
            token=jwt_token,
            market_hub=config["market_hub"],
            base_url=config["base_url"],
            account=config["account"],
            contract=config["contract"],
            size=config["size"],
            timeframe_minutes=config["timeframe"],
            strategy=strategy,
            entry_conf=config["entry_conf"],
            adx_thresh=config["adx_thresh"],
            stop_atr=config["stop_atr"],
            target_atr=config["target_atr"],
            enable_trailing_stop=config["enable_trailing_stop"]
        )
        
        asyncio.run(bot.run())
        
    except KeyboardInterrupt:
        print("\n👋 Bot stopped by user.")
    except Exception as e:
        logging.exception(f"\n❌ A critical error occurred: {e}")        


if __name__ == "__main__":
    main()