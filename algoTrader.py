#!/usr/bin/env python3
"""
Real-Time Trading Bot with Pluggable Strategies

This version supports multiple AI strategies through a clean strategy pattern.
Each strategy handles its own features, predictions, and entry logic.

Now includes simulation/backtesting mode via SimulationBot.

Requirements:
    pip install onnxruntime pandas pandas-ta signalrcore requests numpy scikit-learn python-dateutil
"""

import asyncio
import argparse
import logging

from bot_utils import setup_logging, authenticate, MARKET_HUB, BASE_URL
from config_loader import load_config, merge_config_with_args, validate_config
from strategy_factory import StrategyFactory
from trading_bot import RealTimeBot
from simulation_bot import SimulationBot
import warnings

warnings.filterwarnings('ignore')

# =========================================================
# MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description='Real-Time AI Futures Trading Bot with Pluggable Strategies',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=f"""
Available Strategies: {', '.join(StrategyFactory.list_strategies())}

Example Usage (Live Trading):
  python %(prog)s --account ACC123 --contract CON.F.US.RTY.Z25 --size 1 \\
                  --username USER --apikey KEY --timeframe 5 \\
                  --strategy squeeze_v3 \\
                  --model models/squeeze_v3_model.onnx \\
                  --scaler models/squeeze_v3_scalers.pkl \\
                  --entry_conf 0.55 --adx_thresh 25 --stop_atr 2.0 --target_atr 3.0

Example Usage (Backtesting):
  python %(prog)s --backtest --backtest_data data/historical.csv \\
                  --contract RTY --size 1 --timeframe 5 \\
                  --strategy squeeze_v3 \\
                  --model models/squeeze_v3_model.onnx \\
                  --scaler models/squeeze_v3_scalers.pkl \\
                  --entry_conf 0.55 --adx_thresh 25 --stop_atr 2.0 --target_atr 3.0 \\
                  --tick_size 0.1 --profit_target 6000 --max_loss 3000
"""
    )

    # Config file option
    parser.add_argument('--config', type=str, 
                        help='Path to YAML config file. Command-line args override config values.')     
    parser.add_argument('--debug', action='store_true',
                        help='Enable DEBUG level logging (default: INFO)')
    
    # Backtesting options
    parser.add_argument('--backtest', action='store_true',
                        help='Run in backtesting mode using historical CSV data')
    parser.add_argument('--backtest_data', type=str,
                        help='Path to CSV file with historical OHLCV data for backtesting')
    parser.add_argument('--tick_size', type=float, default=0.01,
                        help='Contract tick size (for backtesting calculations)')
    parser.add_argument('--profit_target', type=float, default=6000,
                        help='Profit target in dollars for backtesting (default: 6000)')
    parser.add_argument('--max_loss', type=float, default=3000,
                        help='Maximum loss limit in dollars for backtesting (default: 3000)')    
    parser.add_argument('--simulation-days', type=int, 
                        help='(Backtest Only) Limit the backtest to the first N days of the CSV data.'
    )
    
    # Account & Contract
    parser.add_argument('--account', type=str,
                        help='TopstepX account ID (not required for backtesting)')
    parser.add_argument('--contract', type=str,
                        help='Full contract ID (e.g., CON.F.US.ENQ.Z25) or symbol for backtesting')
    parser.add_argument('--size', type=int, default=1,
                        help='Trade size (number of contracts)')
    parser.add_argument('--username', type=str,
                        help='TopstepX username (not required for backtesting)')
    parser.add_argument('--apikey', type=str,
                        help='TopstepX API key (not required for backtesting)')
    parser.add_argument('--timeframe', type=int, choices=[1, 3, 5], default=3,
                        help='Bar timeframe in minutes (default: 3)')    
    
    # RealTime Market URL and Base URL
    parser.add_argument('--market_hub', type=str, default=MARKET_HUB,
                        help='Market Hub URL (not required for backtesting)')
    parser.add_argument('--base_url', type=str, default=BASE_URL,
                        help='ProjectX Base URL (not required for backtesting)')
    
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
                        help='Min AI confidence to enter (default: 0.60)')
    parser.add_argument('--adx_thresh', type=int, default=20,
                        help='Min ADX value to enter (default: 20)')
    parser.add_argument('--stop_pts', type=float, default=8.0,
                        help='Stop loss in points (default: 8.0)')
    parser.add_argument('--target_pts', type=float, default=2.0,
                        help='Profit target in points (default: 16.0)')
    parser.add_argument('--enable_trailing_stop', action='store_true',
                        help='Enable trailing stop vs stop order')
    
    # Strategy-specific parameters
    parser.add_argument('--pivot_lookback', type=int, default=8,
                        help='Pivot lookback period (for pivot_reversal strategy)')
    
    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(level=log_level, log_file="bot_log.log")    
    logging.info(f"Logging level set to: {'DEBUG' if args.debug else 'INFO'}")

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
        
        # Validate required fields based on mode
        try:
            if config.get('backtest'):
                # Backtesting mode - only require certain fields
                if not config.get('backtest_data'):
                    raise ValueError("--backtest_data is required when using --backtest")
                if not config.get('contract'):
                    raise ValueError("--contract is required")
            else:
                # Live trading mode - validate all required fields
                validate_config(config)
        except ValueError as e:
            logging.error(f"Configuration error: {e}")
            parser.print_help()
            return
    
    # Log config data for debugging
    logging.info(config)

    # Check mode
    if config.get('backtest'):
        run_backtesting(config)
    else:
        run_live_trading(config)


def run_backtesting(config):
    """Run in backtesting mode using SimulationBot"""
    print("\n" + "="*60)
    print("🔬 BACKTESTING MODE")
    print("="*60 + "\n")
    
    try:
        # Create strategy
        strategy_kwargs = {}
        if config['strategy'] == '3min_pivot_reversal' or config['strategy'] == '5min_pivot_reversal':
            strategy_kwargs['pivot_lookback'] = config.get("pivot_lookback", 8)
        
        strategy = StrategyFactory.create_strategy(
            strategy_name=config["strategy"],
            model_path=config["model"],
            scaler_path=config["scaler"],
            contract_symbol=config["contract"],  # Use contract as symbol directly in backtest
            **strategy_kwargs
        )
        
        logging.info(f"✅ Strategy '{config['strategy']}' created successfully!")
        
        # Create simulation bot
        bot = SimulationBot(
            csv_path=config["backtest_data"],
            contract=config["contract"],
            size=config["size"],
            timeframe_minutes=config["timeframe"],
            strategy=strategy,
            entry_conf=config["entry_conf"],
            adx_thresh=config["adx_thresh"],
            stop_pts=config["stop_pts"],
            target_pts=config["target_pts"],
            tick_size=config.get("tick_size", 0.01),
            profit_target=config.get("profit_target", 6000),
            max_loss_limit=config.get("max_loss", 3000),
            enable_trailing_stop=config.get("enable_trailing_stop", False),
            simulation_days=config.get("simulation_days"),
        )
        
        # Run simulation
        asyncio.run(bot.run())
        
    except KeyboardInterrupt:
        print("\n👋 Backtesting stopped by user.")
    except Exception as e:
        logging.exception(f"\n❌ A critical error occurred: {e}")


def run_live_trading(config):
    """Run in live trading mode using RealTimeBot"""
    print("\n" + "="*60)
    print("📡 LIVE TRADING MODE")
    print("="*60 + "\n")
    
    # Authenticate
    jwt_token = authenticate(config["base_url"], config["username"], config["apikey"])
    if not jwt_token:
        return
    
    logging.info(f"🎫 Token received. Creating strategy...")
    
    try:
        # Create strategy
        strategy_kwargs = {}
        if config['strategy'] == '3min_pivot_reversal' or config['strategy'] == '5min_pivot_reversal':
            strategy_kwargs['pivot_lookback'] = config.get("pivot_lookback", 8)
        
        strategy = StrategyFactory.create_strategy(
            strategy_name=config["strategy"],
            model_path=config["model"],
            scaler_path=config["scaler"],
            contract_symbol=None,
            **strategy_kwargs
        )
        
        logging.info(f"✅ Strategy '{config['strategy']}' created successfully!")
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
            stop_pts=config["stop_pts"],
            target_pts=config["target_pts"],
            enable_trailing_stop=config.get("enable_trailing_stop", False)
        )
        
        asyncio.run(bot.run())
        
    except KeyboardInterrupt:
        print("\n👋 Bot stopped by user.")
    except Exception as e:
        logging.exception(f"\n❌ A critical error occurred: {e}")


if __name__ == "__main__":
    main()