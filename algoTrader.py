#!/usr/bin/env python3
"""
Real-Time Trading Bot with Pluggable Strategies

This version supports multiple AI strategies through a clean strategy pattern.
Each strategy handles its own features, predictions, and entry logic.

Requirements:
    pip install onnxruntime pandas pandas-ta signalrcore requests numpy scikit-learn python-dateutil
"""

import numpy as np
import asyncio
import argparse
import requests
from pysignalr.client import SignalRClient
from datetime import datetime, timedelta, timezone
from config_loader import load_config, merge_config_with_args, validate_config
from collections import deque
import warnings
import pandas as pd

import logging

from strategy_factory import StrategyFactory
from strategy_base import BaseStrategy

warnings.filterwarnings('ignore')

MARKET_HUB = "https://rtc.topstepx.com/hubs/market"
BASE_URL = "https://api.topstepx.com/api"

CONTRACTS = {
    "CON.F.US.ENQ.Z25": 'NQ',
    "CON.F.US.EP.Z25": 'ES',
    "CON.F.US.YM.Z25": 'YM',
    "CON.F.US.RTY.Z25": 'RTY'
}

TICK_INFO = {
    'ES': {'tick_size': 0.25},
    'NQ': {'tick_size': 0.25},
    'YM': {'tick_size': 1.0},
    'RTY': {'tick_size': 0.1}
}

# =========================================================
# LOGGING SETUP
# =========================================================
def setup_logging(level=logging.INFO, log_file=None):
    """Configures basic logging, prioritizing file if specified."""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    handlers = [] # Start with no handlers

    if log_file:
        print(f"Logging configured to file: {log_file}") # Still print confirmation to console
        handlers.append(logging.FileHandler(log_file))
    else:
        # Fallback to console only if no log file is given
        print("Logging configured to console.")
        handlers.append(logging.StreamHandler())

    # If no handlers were added (e.g., log_file was empty string), add console handler
    if not handlers:
         handlers.append(logging.StreamHandler())

    logging.basicConfig(level=level, format=log_format, handlers=handlers, force=True) # Use force=True to allow reconfiguration
    logging.info("--- Log Start ---") # Add a marker for new log session

# =========================================================
# AUTHENTICATION
# =========================================================
def authenticate(base_url, username, api_key):
    """Authenticates and returns a JWT token."""
    auth_url = f"{base_url}/Auth/loginKey"
    payload = {"userName": username, "apiKey": api_key}
    try:
        logging.info("🔐 Authenticating...")
        logging.info(payload)     
        response = requests.post(auth_url, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get('success') and data.get('token'):
            logging.info("✅ Authentication successful!")
            return data['token']
        else:
            logging.error(f"❌ Authentication failed: {data.get('errorMessage', 'Unknown error')}")
            return None
    except Exception as e:
        logging.exception(f"❌ Authentication error: {e}")
        return None


# =========================================================
# REAL-TIME TRADING BOT CLASS
# =========================================================
class RealTimeBot:
    def __init__(
        self, 
        token, 
        market_hub,
        base_url,
        account, 
        contract, 
        size, 
        timeframe_minutes,
        strategy: BaseStrategy,
        entry_conf, 
        adx_thresh, 
        stop_atr, 
        target_atr,
        enable_trailing_stop=False
    ):
        """
        Initialize the trading bot with a strategy.
        
        Args:
            token: Authentication token
            account: Trading account ID
            contract: Contract ID
            size: Position size
            timeframe_minutes: Bar timeframe
            strategy: Strategy instance (implements BaseStrategy)
            entry_conf: Minimum confidence for entry
            adx_thresh: Minimum ADX for entry
            stop_atr: Stop loss ATR multiplier
            target_atr: Profit target ATR multiplier
            enable_trailing_stop: Enable trailing stops
        """        
        self.hub_url = f"{market_hub}?access_token={token}"
        self.base_url = base_url
        self.account = account
        self.contract = contract
        self.size = size
        self.timeframe_minutes = int(timeframe_minutes)
        self.client = SignalRClient(self.hub_url)
        self.token = token
        self.enable_trailing_stop = enable_trailing_stop
        
        self.current_bar = {}
        self.current_bar_time = None
        self.bar_lock = asyncio.Lock()
        self.closer_task = None
        
        # State Management
        self.in_position = False
        self.position_type = None
        self.entry_price = None
        self.stop_loss = None
        self.profit_target = None
        self.stop_orderId = None
        self.limit_orderId = None
        
        # Strategy
        self.strategy = strategy
        self.contract_symbol = CONTRACTS.get(self.contract, "ES")
        
        # Historical bars (strategy determines how many needed)
        seq_len = self.strategy.get_sequence_length()
        self.num_historical_candles_needed = seq_len
        self.historical_bars = deque(maxlen=seq_len)
        
        # Trading parameters
        self.entry_conf = entry_conf
        self.adx_thresh = adx_thresh
        self.stop_atr_mult = stop_atr
        self.target_atr_mult = target_atr
        
        print(f"🤖 Bot initialized for {self.contract} ({self.contract_symbol}) "
              f"on {self.timeframe_minutes}-min timeframe.")
        print(f"📈 Trade Params: Entry={self.entry_conf}, ADX={self.adx_thresh}, "
              f"Stop={self.stop_atr_mult} ATR, Target={self.target_atr_mult} ATR")
        logging.info(f"📊 Strategy: {self.strategy.__class__.__name__}")        
        logging.info(f"📈 Trade Params: Entry={self.entry_conf}, ADX={self.adx_thresh}, "
              f"Stop={self.stop_atr_mult} ATR, Target={self.target_atr_mult} ATR")
        
        # Load strategy model and scaler
        self.strategy.load_model()
        self.strategy.load_scaler()
        
        # Register handlers
        self.client.on_open(self.on_open)
        self.client.on_close(self.on_close)
        self.client.on_error(self.on_error)
        self.client.on("GatewayTrade", self.process_tick)

    # =========================================================
    # CONNECTION HANDLERS
    # =========================================================
    async def on_open(self):
        """Callback when connection opens - ORIGINAL METHOD."""
        print("✅ Connected to market hub")
        try:
            await self.client.send("SubscribeContractTrades", [self.contract])
            logging.info(f"✅ Subscription successful for {self.contract}")
        except Exception as e:
            logging.exception(f"❌ Subscription error: {e}")

    async def on_close(self):
        """Callback when connection closes."""
        print('🔌 Disconnected from the server')
        if self.closer_task:
            self.closer_task.cancel()

    async def on_error(self, message):
        """Callback on error."""
        logging.exception(f"❌ SignalR Error: {message}")

    # =========================================================
    # HISTORICAL DATA FETCHING
    # =========================================================
    async def fetch_historical_data(self):
        """Fetches recent bars to prime the historical data deque - ORIGINAL METHOD."""
        historical_url = f"{self.base_url}/History/retrieveBars"
        end_time_dt = datetime.now(timezone.utc).replace(microsecond=0)
        # Fetch a bit more history (e.g., 3 days) to ensure enough warmup for indicators        
        duration_minutes = self.strategy.get_sequence_length() * self.timeframe_minutes
        buffer_minutes = 5 * self.timeframe_minutes # Optional safety buffer
        start_time_dt = end_time_dt - timedelta(minutes=(duration_minutes + buffer_minutes))
        payload = {
            "contractId": self.contract,
            "live": False,
            "startTime": start_time_dt.isoformat().replace('+00:00', 'Z'),
            "endTime": end_time_dt.isoformat().replace('+00:00', 'Z'),
            "unit": 2,
            "unitNumber": self.timeframe_minutes,
            "limit": self.historical_bars.maxlen,
            "includePartialBar": False
        }
        headers = {'Authorization': f'Bearer {self.token}'}
        try:
            response = requests.post(historical_url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            bars_fetched = 0
            for bar in response.json().get('bars', []):
                self.historical_bars.append({
                    "timestamp": datetime.fromisoformat(bar['t']).isoformat(),
                    "open": bar['o'],
                    "high": bar['h'],
                    "low": bar['l'],
                    "close": bar['c'],
                    "volume": bar['v']
                })
                bars_fetched += 1
            print(f"✅ Successfully pre-filled {bars_fetched} historical bars.")
            if bars_fetched < self.num_historical_candles_needed:
                print(f"⚠️ Warning: Fetched fewer bars ({bars_fetched}) than needed "
                      f"({self.num_historical_candles_needed}) for full AI warmup.")
        except Exception as e:
            logging.exception(f"❌ Could not fetch historical data: {e}.")

    # =========================================================
    # TICK PROCESSING
    # =========================================================
    async def process_tick(self, data):
        """Processes incoming tick data - ORIGINAL METHOD."""
        try:
            # Handle list format: ['CONTRACT', [ {...}, {...} ]]
            if isinstance(data, list) and len(data) >= 2 and isinstance(data[1], list):
                trades = data[1]
            # Handle single dict format: { ... }
            elif isinstance(data, dict):
                trades = [data]
            else:
                trades = []

            for trade in trades:
                await self.handle_trade(trade)
        except Exception as e:
            logging.exception(f"❌ process_tick error: {e} | Data: {data}")

    # =========================================================
    # BAR AGGREGATION & AI PREDICTION
    # =========================================================
    def _get_bar_time(self, ts):
        """Get the bar time for a given timestamp."""
        ts = ts.replace(second=0, microsecond=0)
        minutes = (ts.minute // self.timeframe_minutes) * self.timeframe_minutes
        return ts.replace(minute=minutes)

    async def _close_and_print_bar(self):
        """Close current bar and run AI prediction."""
        if not self.current_bar:
            return
        
        # Add to historical bars
        self.historical_bars.append(self.current_bar)
        
        bar_time_str = datetime.fromisoformat(
            self.current_bar["timestamp"]
        ).strftime("%Y-%m-%d %H:%M")
        
        print(f"📊 Bar: {bar_time_str} | "
              f"O:{self.current_bar['open']:.2f} "
              f"H:{self.current_bar['high']:.2f} "
              f"L:{self.current_bar['low']:.2f} "
              f"C:{self.current_bar['close']:.2f} "
              f"V:{self.current_bar['volume']}")
        
        logging.info(f"📊 Bar: {bar_time_str} | "
              f"O:{self.current_bar['open']:.2f} "
              f"H:{self.current_bar['high']:.2f} "
              f"L:{self.current_bar['low']:.2f} "
              f"C:{self.current_bar['close']:.2f} "
              f"V:{self.current_bar['volume']}")
        
        # Run AI if enough bars
        if len(self.historical_bars) >= self.num_historical_candles_needed:
            await self._run_ai_prediction()
        else:
            logging.info(f"⏳ Waiting for more bars... "
                  f"({len(self.historical_bars)}/{self.num_historical_candles_needed})")
        
        # Reset bar state
        self.current_bar, self.current_bar_time = {}, None

    async def _run_ai_prediction(self):
        """Run AI prediction using the strategy"""
        if self.in_position:
            return
        
        try:
            # Convert historical bars to DataFrame
            df = pd.DataFrame(list(self.historical_bars))
            
            # Add strategy-specific features
            df = self.strategy.add_features(df)
            
            # Validate features
            if not self.strategy.validate_features(df):
                logging.exception("❌ Feature validation failed")
                return
            
            # Get prediction from strategy
            prediction, confidence = self.strategy.predict(df)
            
            # Get latest bar for entry checks
            latest_bar = df.iloc[-1].to_dict()
            
            # Check if should enter trade
            should_enter, direction = self.strategy.should_enter_trade(
                prediction,
                confidence,
                latest_bar,
                self.entry_conf,
                self.adx_thresh
            )
            
            # Display prediction
            pred_labels = {0: "HOLD", 1: "BUY", 2: "SELL"}
            print(f"🤖 AI: {pred_labels[prediction]} (Conf: {confidence:.2%}) | "
                  f"ADX: {latest_bar.get('adx', 0):.1f}")
            logging.info(f"AI: {pred_labels[prediction]} (Conf: {confidence:.2%}) ADX: {latest_bar.get('adx', 0):.1f}")
            
            if should_enter:
                # ORIGINAL ENTRY LOGIC PRESERVED
                close_price = latest_bar['close']
                atr = latest_bar.get('atr', 0)
                
                if atr <= 0:
                    logging.exception("❌ Invalid ATR, skipping entry")
                    return
                
                tick_size = TICK_INFO[self.contract_symbol]['tick_size']
                
                if direction == 'LONG':
                    self.in_position = True
                    self.position_type = 'LONG'
                    self.entry_price = close_price
                    self.stop_loss = self.entry_price - (atr * self.stop_atr_mult)
                    self.profit_target = self.entry_price + (atr * self.target_atr_mult)
                    
                    print("="*40)
                    print(f"🔥🔥🔥 ENTERING LONG @ {self.entry_price:.2f} 🔥🔥🔥")
                    print(f"  SL: {self.stop_loss:.2f} | PT: {self.profit_target:.2f}")
                    print("="*40)
                    logging.info(f"LONG @ {self.entry_price:.2f} SL: {self.stop_loss:.2f} | PT: {self.profit_target:.2f}")
                    
                    # Calculate ticks                    
                    stop_loss_ticks = int((self.entry_price - self.stop_loss) / tick_size)                    
                    take_profit_ticks = int((self.profit_target - self.entry_price) / tick_size)
                    
                    # Place order with ORIGINAL parameters
                    await self._place_order(0, stop_ticks=stop_loss_ticks, take_profit_ticks=take_profit_ticks)
                    
                else:  # SHORT
                    self.in_position = True
                    self.position_type = 'SHORT'
                    self.entry_price = close_price
                    self.stop_loss = self.entry_price + (atr * self.stop_atr_mult)
                    self.profit_target = self.entry_price - (atr * self.target_atr_mult)
                    
                    print("="*40)
                    print(f"🥶🥶🥶 ENTERING SHORT @ {self.entry_price:.2f} 🥶🥶🥶")
                    print(f"  SL: {self.stop_loss:.2f} | PT: {self.profit_target:.2f}")
                    print("="*40)
                    logging.info(f"SHORT @ {self.entry_price:.2f} SL: {self.stop_loss:.2f} | PT: {self.profit_target:.2f}")
                    
                    # Calculate ticks (ORIGINAL CALCULATION)                    
                    stop_loss_ticks = int((self.stop_loss - self.entry_price) / tick_size)
                    take_profit_ticks = int((self.profit_target - self.entry_price) / tick_size) 
                    
                    # Place order with ORIGINAL parameters
                    await self._place_order(1, stop_ticks=stop_loss_ticks, take_profit_ticks=take_profit_ticks)
                
        except Exception as e:
            logging.exception(f"❌ Error during AI prediction: {e}")            

    # =========================================================
    # ORDER MANAGEMENT
    # =========================================================
    async def _place_order(self, side, type=2, stop_ticks=10, take_profit_ticks=20):
        """Enter long/short position"""
        order_url = f"{self.base_url}/Order/place"
        payload = {
            "accountId": self.account,
            "contractId": self.contract,
            "type": type,  # Market order
            "side": side,  # 0 = Bid (buy), 1 = Ask (sell)
            "size": self.size,  # Size of the order
            "stopLossBracket": {
                "ticks": stop_ticks,
                "type": 5 if self.enable_trailing_stop else 4
            },
            "takeProfitBracket": {
                "ticks": take_profit_ticks,
                "type": 1
            }
        }        
        headers = {'Authorization': f'Bearer {self.token}'}
        try:
            response = requests.post(order_url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()            
            if data.get('success') and data.get('orderId'):
                logging.info(f"✅ Order placed successfully: {data.get('orderId')}")
                return data.get("orderId")
            else:
                logging.exception(f"❌ Order failed: {data.get('errorMessage')}")
                return None
        except Exception as e:
            logging.exception(f"❌ Could not place order: {e}.")
            return None

    # =========================================================
    # BAR CLOSER WATCHER
    # =========================================================
    async def bar_closer_watcher(self):
        """Background task to watch the clock and force-close bars."""
        logging.info("⏳ Bar closer watcher started...")
        while True:
            try:
                if not self.current_bar_time:
                    await asyncio.sleep(0.1)
                    continue
                
                next_bar_time = self.current_bar_time + timedelta(
                    minutes=self.timeframe_minutes
                )
                sleep_duration = (
                    next_bar_time - datetime.now(timezone.utc)
                ).total_seconds()
                
                if sleep_duration > 0:
                    await asyncio.sleep(sleep_duration + 0.05)
                
                async with self.bar_lock:
                    if self.current_bar and self.current_bar_time < next_bar_time:
                        await self._close_and_print_bar()
                        
            except asyncio.CancelledError:
                logging.info("Bar closer watcher stopping.")
                break
            except Exception as e:
                logging.exception(f"Error in bar_closer_watcher: {e}")
                await asyncio.sleep(1)

    # =========================================================
    # TRADE HANDLING
    # =========================================================
    async def handle_trade(self, trade):
        """Aggregates ticks into bars and checks for exits on every tick."""
        try:
            ts = datetime.fromisoformat(trade.get("timestamp"))
            price = trade.get("price")
            volume = trade.get("volume", 0)
            
            if price is None:
                return
            
            # Check exits if in position
            if self.in_position:
                exit_price, exit_reason = None, None
                
                if self.stop_loss is None or self.profit_target is None:
                    logging.error("⚠️ Exit check skipped: stop_loss or profit_target not set.")
                elif self.position_type == 'LONG':
                    if price <= self.stop_loss:
                        exit_price, exit_reason = self.stop_loss, 'STOP_LOSS'
                    elif price >= self.profit_target:
                        exit_price, exit_reason = self.profit_target, 'PROFIT_TARGET'
                elif self.position_type == 'SHORT':
                    if price >= self.stop_loss:
                        exit_price, exit_reason = self.stop_loss, 'STOP_LOSS'
                    elif price <= self.profit_target:
                        exit_price, exit_reason = self.profit_target, 'PROFIT_TARGET'
                
                if exit_reason:
                    pnl = ((exit_price - self.entry_price) if self.position_type == 'LONG'
                           else (self.entry_price - exit_price))
                    print("="*40)
                    print(f"🛑 EXIT {self.position_type} @ {exit_price:.2f} ({exit_reason})")
                    print(f"  Entry: {self.entry_price:.2f} | PnL Points: {pnl:.2f}")
                    print("="*40)
                    logging.info(f"EXIT {self.position_type} @ {exit_price:.2f} ({exit_reason}) Entry: {self.entry_price:.2f} | PnL Points: {pnl:.2f}")
                    
                    self.in_position = False
                    self.position_type = None
                    self.entry_price = None
                    self.stop_loss = None
                    self.profit_target = None
            
            # Bar aggregation
            bar_time = self._get_bar_time(ts)
            async with self.bar_lock:
                if bar_time != self.current_bar_time:
                    if self.current_bar:
                        await self._close_and_print_bar()
                    self.current_bar_time = bar_time
                    self.current_bar = {
                        "timestamp": bar_time.isoformat(),
                        "open": price,
                        "high": price,
                        "low": price,
                        "close": price,
                        "volume": volume
                    }
                elif self.current_bar:
                    self.current_bar["high"] = max(self.current_bar.get("high", price), price)
                    self.current_bar["low"] = min(self.current_bar.get("low", price), price)
                    self.current_bar["close"] = price
                    self.current_bar["volume"] = self.current_bar.get("volume", 0) + volume
                else:
                    self.current_bar_time = bar_time
                    self.current_bar = {
                        "timestamp": bar_time.isoformat(),
                        "open": price,
                        "high": price,
                        "low": price,
                        "close": price,
                        "volume": volume
                    }
                    
        except Exception as e:
            logging.exception(f"❌ handle_trade error: {e} | Trade: {trade}")

    # =========================================================
    # MAIN RUN LOOP (ORIGINAL SEQUENCE)
    # =========================================================
    async def run(self):
        """Starts the bot - ORIGINAL SEQUENCE."""
        await self.fetch_historical_data()
        print("🚀 Starting bot connection...")
        self.closer_task = asyncio.create_task(self.bar_closer_watcher())
        await self.client.run()


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
                        help='Min AI confidence to enter (default: 0.60)')
    parser.add_argument('--adx_thresh', type=int, default=20,
                        help='Min ADX value to enter (default: 20)')
    parser.add_argument('--stop_atr', type=float, default=1.5,
                        help='Stop loss multiplier (x ATR) (default: 1.5)')
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
    
    # Authenticate
    jwt_token = authenticate(config["base_url"], config["username"], config["apikey"])
    if not jwt_token:
        return
    
    logging.info(f"🎫 Token received. Creating strategy...")
    
    try:
        # Get contract symbol
        contract_symbol = CONTRACTS.get(args.contract, "ES")
        
        # Create strategy
        strategy_kwargs = {}
        if args.strategy == '3min_pivot_reversal' or args.strategy == '5min_pivot_reversal':
            strategy_kwargs['pivot_lookback'] = config["pivot_lookback"]
        
        strategy = StrategyFactory.create_strategy(
            strategy_name=config["strategy"],
            model_path=config["model"],
            scaler_path=config["scaler"],
            contract_symbol=contract_symbol,
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