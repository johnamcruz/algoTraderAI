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
import os
import sys
from datetime import datetime

from utils.bot_utils import setup_logging, authenticate, MARKET_HUB, BASE_URL
from utils.config_loader import load_config, merge_config_with_args, validate_config
from strategies.strategy_factory import StrategyFactory
from bots.trading_bot import RealTimeBot
from bots.simulation_bot import SimulationBot
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
  python %(prog)s --account ACC123 --contract CON.F.US.MES.M26 --username USER --apikey KEY \\
                  --strategy cisd-ote7 --model models/cisd_ote_hybrid_v7.onnx \\
                  --risk_amount 200 --entry_conf 0.80

Example Usage (Backtesting):
  python %(prog)s --backtest --backtest_data data/ES_continuous_5min.csv \\
                  --contract CON.F.US.MES.M26 --timeframe 5 \\
                  --strategy cisd-ote7 --model models/cisd_ote_hybrid_v7.onnx \\
                  --risk_amount 200 --entry_conf 0.80 \\
                  --tick_size 0.25 --profit_target 12000 --max_loss 400
"""
    )

    # Config file option
    parser.add_argument('--config', type=str, 
                        help='Path to YAML config file. Command-line args override config values.')     
    parser.add_argument('--debug', action='store_true', default=None,
                        help='Enable DEBUG level logging (default: INFO)')
    parser.add_argument('--quiet', action='store_true', default=None,
                        help='Suppress all output except the final simulation results summary')

    # Backtesting options
    parser.add_argument('--backtest', action='store_true', default=None,
                        help='Run in backtesting mode using historical CSV data')
    parser.add_argument('--backtest_data', type=str,
                        help='Path to CSV file with historical OHLCV data for backtesting')
    parser.add_argument('--tick_size', type=float, default=None,
                        help='Contract tick size (for backtesting calculations)')
    parser.add_argument('--profit_target', type=float, default=None,
                        help='Profit target in dollars for backtesting (default: 6000)')
    parser.add_argument('--no-profit-target', action='store_true', default=None,
                        help='Disable session profit target cap — run full regime window')
    parser.add_argument('--max_loss', type=float, default=None,
                        help='Maximum loss limit in dollars for backtesting (default: 3000)')    
    parser.add_argument('--simulation-days', type=int,
                        help='(Backtest Only) Limit the backtest to the last N days of the CSV data.'
    )
    parser.add_argument('--start-date', type=str,
                        help='(Backtest Only) Start date for simulation (YYYY-MM-DD). Overrides --simulation-days.')
    parser.add_argument('--end-date', type=str,
                        help='(Backtest Only) End date for simulation (YYYY-MM-DD). Defaults to end of CSV if omitted.')
    
    # Account & Contract
    parser.add_argument('--account', type=str,
                        help='TopstepX account ID (not required for backtesting)')
    parser.add_argument('--contract', type=str,
                        help='Full contract ID (e.g., CON.F.US.ENQ.Z25) or symbol for backtesting')
    parser.add_argument('--size', type=int, default=None,
                        help='Trade size (number of contracts)')
    parser.add_argument('--risk_amount', type=float, default=None,
                        help='Max dollars to risk per trade for dynamic sizing (overrides --size when set)')
    parser.add_argument('--username', type=str,
                        help='TopstepX username (not required for backtesting)')
    parser.add_argument('--apikey', type=str,
                        help='TopstepX API key (not required for backtesting)')
    parser.add_argument('--timeframe', type=int, choices=[1, 3, 5], default=None,
                        help='Bar timeframe in minutes (default: 5)')    
    
    # RealTime Market URL and Base URL
    parser.add_argument('--market_hub', type=str, default=None,
                        help='Market Hub URL (not required for backtesting)')
    parser.add_argument('--base_url', type=str, default=None,
                        help='ProjectX Base URL (not required for backtesting)')
    
    # Strategy Selection
    parser.add_argument('--strategy', type=str, default=None,
                        choices=StrategyFactory.list_strategies(),
                        help='Strategy to use')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to the ONNX model file (.onnx)')
    
    # Trading Parameters
    parser.add_argument('--entry_conf', type=float, default=None,
                        help='Min AI confidence to enter (default: 0.9)')
    parser.add_argument('--adx_thresh', type=int, default=None,
                        help='Min ADX value to enter (default: 0)')
    parser.add_argument('--stop_pts', type=float, default=None,
                        help='Stop loss in points (optional if strategy provides its own)')
    parser.add_argument('--target_pts', type=float, default=None,
                        help='Profit target in points (optional if strategy provides its own)')
    parser.add_argument('--enable_trailing_stop', action='store_true', default=None,
                        help='Use TopstepX native trailing stop bracket (type 5) — trails price continuously')
    parser.add_argument('--breakeven_on_2r', action=argparse.BooleanOptionalAction, default=None,
                        help='Move stop to entry price once trade reaches 1R profit (bot-managed, works in sim). '
                             'Default: on. Use --no-breakeven_on_2r to disable.')
    parser.add_argument('--high_conf_multiplier', type=float, default=None,
                        help='Scale risk_amount by this factor when confidence ≥0.90 (e.g. 2.0 doubles size)')
    parser.add_argument('--max_contracts', type=int, default=None,
                        help='Maximum contracts per trade regardless of risk sizing (default: 15)')
    parser.add_argument('--min_stop_pts', type=float, default=None,
                        help='Minimum stop distance in points — signals with tighter stops are skipped (default: 1.0)')
    parser.add_argument('--min_stop_atr', type=float, default=None,
                        help='Dynamic minimum stop as a multiple of ATR14 (default: 0.5). '
                             'Stop must be ≥ this × ATR14; effective floor is max(--min_stop_pts, mult×ATR). '
                             'Calibrated for MES/MNQ/MGC on 5-min bars. Set 0 to disable.')
    # Strategy-specific parameters
    parser.add_argument('--pivot_lookback', type=int, default=None,
                        help='Pivot lookback period (for pivot_reversal strategy)')
    parser.add_argument('--min_vty_regime', type=float, default=None,
                        help='(cisd-ote) Regime gate: skip trades when vty_regime (atr14/atr_ma50) '
                             'is below this value (0.0=disabled, 0.8=block when vol is 20%% below '
                             'its 50-bar average). Persistent across sustained low-vol periods.')
    parser.add_argument('--min_entry_distance', type=float, default=None,
                        help='(cisd-ote) OTE depth gate: skip signals where entry_distance_pct is '
                             'below this value (0.0=disabled, 3.0=recommended). Filters shallow '
                             'zone touches; winners avg 3.9-4.5 vs losers 2.1-2.9 in backtests.')
    parser.add_argument('--min_risk_rr', type=float, default=None,
                        help='(cisd-ote7) RR gate: skip trades when model predicted_rr is below '
                             'this value (0.0=disabled, 2.0=recommended). F4 calibration: '
                             'predict>=2.0 gives 81%% 3R hit rate on val signals.')
    args = parser.parse_args()

    # Load configuration first so we know the contract for the log filename
    if args.config:
        try:
            config = load_config(args.config)
            config = merge_config_with_args(config, args)
            validate_config(config)
        except (FileNotFoundError, ValueError, ImportError) as e:
            print(f"Configuration error: {e}")
            return
    else:
        config = merge_config_with_args({}, args)
        config.pop('config', None)

    if args.no_profit_target:
        config['profit_target'] = None

        try:
            if config.get('backtest'):
                if not config.get('backtest_data'):
                    raise ValueError("--backtest_data is required when using --backtest")
                if not config.get('contract'):
                    raise ValueError("--contract is required")
            else:
                validate_config(config)
        except ValueError as e:
            print(f"Configuration error: {e}")
            parser.print_help()
            return

    # Setup logging — include instrument symbol so multi-bot runs get separate files
    log_level = logging.DEBUG if args.debug else logging.INFO
    os.makedirs("logs", exist_ok=True)
    contract = config.get('contract', 'unknown')
    symbol = contract.split('.')[-2] if '.' in contract else contract
    mode = "backtest" if config.get('backtest') else "live"
    log_file = f"logs/bot_{symbol}_{mode}_{datetime.now().strftime('%Y%m%d')}.log"
    setup_logging(level=log_level, log_file=log_file)
    logging.info("--- Log Start ---")
    logging.info(f"Logging level set to: {'DEBUG' if args.debug else 'INFO'}")
    if args.config:
        logging.info(f"Loaded config from: {args.config}")
    
    # Log config data for debugging (credentials redacted)
    safe_config = {k: '***' if k in ('apikey', 'password', 'token') else v for k, v in config.items()}
    logging.info(safe_config)

    # Check mode
    if config.get('backtest'):
        run_backtesting(config)
    else:
        run_live_trading(config)


def run_backtesting(config):
    """Run in backtesting mode using SimulationBot"""
    quiet = config.get('quiet', False)

    if not quiet:
        print("\n" + "="*60)
        print("🔬 BACKTESTING MODE")
        print("="*60 + "\n")

    try:
        # Create strategy — only pass min_risk_rr when explicitly provided so each
        # strategy's constructor default (e.g. vwap=4.0, others=0.0) is respected.
        strategy_kwargs = {}
        if 'min_risk_rr' in config:
            strategy_kwargs['min_risk_rr'] = config['min_risk_rr']
        if config['strategy'] == '3min_pivot_reversal' or config['strategy'] == '5min_pivot_reversal':
            strategy_kwargs['pivot_lookback'] = config.get("pivot_lookback", 8)
        if config['strategy'] == 'cisd-ote':
            strategy_kwargs['min_vty_regime']     = config.get('min_vty_regime', 0.75)
            strategy_kwargs['min_entry_distance'] = config.get('min_entry_distance', 3.0)

        strategy = StrategyFactory.create_strategy(
            strategy_name=config["strategy"],
            model_path=config["model"],

            contract_symbol=config["contract"],
            **strategy_kwargs
        )

        # cisd-ote7 uses risk-head tier-snapping for TP — high_conf_multiplier
        # would corrupt those calibrated targets, so it is always disabled.
        high_conf_mult = 1.0 if config['strategy'] == 'cisd-ote7' else config.get('high_conf_multiplier', 1.0)

        bot = SimulationBot(
            csv_path=config["backtest_data"],
            contract=config["contract"],
            size=config["size"],
            timeframe_minutes=config["timeframe"],
            strategy=strategy,
            entry_conf=config["entry_conf"],
            adx_thresh=config["adx_thresh"],
            stop_pts=config.get("stop_pts"),
            target_pts=config.get("target_pts"),
            tick_size=config.get("tick_size", 0.01),
            profit_target=config.get("profit_target", 6000),
            max_loss_limit=config.get("max_loss", 3000),
            enable_trailing_stop=config.get("enable_trailing_stop", False),
            simulation_days=config.get("simulation_days"),
            start_date=config.get("start_date"),
            end_date=config.get("end_date"),
            risk_amount=config.get("risk_amount"),
            high_conf_multiplier=high_conf_mult,
            max_contracts=config.get("max_contracts", 15),
            min_stop_pts=config.get("min_stop_pts", 1.0),
            min_stop_atr_mult=config.get("min_stop_atr", 0.5),
            breakeven_on_2r=config.get("breakeven_on_2r", True),
        )

        if quiet:
            # Suppress all output during run; print only the summary at the end
            with open(os.devnull, 'w') as devnull:
                old_stdout = sys.stdout
                sys.stdout = devnull
                try:
                    asyncio.run(bot.run())
                finally:
                    sys.stdout = old_stdout
            bot._print_summary()
        else:
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
        # Create strategy — only pass min_risk_rr when explicitly provided so each
        # strategy's constructor default (e.g. vwap=4.0, others=0.0) is respected.
        strategy_kwargs = {}
        if 'min_risk_rr' in config:
            strategy_kwargs['min_risk_rr'] = config['min_risk_rr']
        if config['strategy'] == '3min_pivot_reversal' or config['strategy'] == '5min_pivot_reversal':
            strategy_kwargs['pivot_lookback'] = config.get("pivot_lookback", 8)
        if config['strategy'] == 'cisd-ote':
            strategy_kwargs['min_vty_regime']     = config.get('min_vty_regime', 0.75)
            strategy_kwargs['min_entry_distance'] = config.get('min_entry_distance', 3.0)

        strategy = StrategyFactory.create_strategy(
            strategy_name=config["strategy"],
            model_path=config["model"],

            contract_symbol=None,
            **strategy_kwargs
        )
        
        logging.info(f"✅ Strategy '{config['strategy']}' created successfully!")
        logging.info(f"Starting bot...")
        
        # cisd-ote7 uses risk-head tier-snapping for TP — high_conf_multiplier
        # would corrupt those calibrated targets, so it is always disabled.
        high_conf_mult = 1.0 if config['strategy'] == 'cisd-ote7' else config.get('high_conf_multiplier', 1.0)

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
            stop_pts=config.get("stop_pts"),
            target_pts=config.get("target_pts"),
            enable_trailing_stop=config.get("enable_trailing_stop", False),
            risk_amount=config.get("risk_amount"),
            high_conf_multiplier=high_conf_mult,
            max_contracts=config.get("max_contracts", 15),
            min_stop_pts=config.get("min_stop_pts", 1.0),
            min_stop_atr_mult=config.get("min_stop_atr", 0.5),
            breakeven_on_2r=config.get("breakeven_on_2r", True),
            username=config.get("username"),
            api_key=config.get("apikey"),
        )

        asyncio.run(bot.run())
        
    except KeyboardInterrupt:
        print("\n👋 Bot stopped by user.")
    except Exception as e:
        logging.exception(f"\n❌ A critical error occurred: {e}")


if __name__ == "__main__":
    main()