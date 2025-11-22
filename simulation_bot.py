#!/usr/bin/env python3
"""
Simulation Bot Engine - for backtesting trading strategies

This bot reads historical OHLCV data from a CSV file and simulates
trading operations to evaluate strategy performance. It:

1.  Loads historical bar data from CSV
2.  Processes bars sequentially 
3.  Calls the pluggable AI strategy on each bar
4.  Simulates trade execution and tracks P&L
5.  Reports results including profit targets and max loss limits
"""

import logging
import pandas as pd
from datetime import datetime
from strategy_base import BaseStrategy
from trading_bot_base import TradingBot

CONTRACTS = {
    "CON.F.US.ENQ.Z25": "NQ",
    "CON.F.US.EP.Z25": "ES",
}


class SimulationBot(TradingBot):
    """Simulation bot for backtesting strategies"""
    
    def __init__(
        self,
        csv_path,
        contract,
        size,
        timeframe_minutes,
        strategy: BaseStrategy,
        entry_conf,
        adx_thresh,
        stop_pts,
        target_pts,
        tick_size=0.01,
        profit_target=6000,
        max_loss_limit=3000,
        enable_trailing_stop=False,
        simulation_days=None
    ):
        """
        Initialize the simulation bot.
        
        Args:
            csv_path: Path to CSV file with OHLCV data
            contract: Contract symbol (for display purposes)
            size: Position size
            timeframe_minutes: Bar timeframe
            strategy: Strategy instance (implements BaseStrategy)
            entry_conf: Minimum confidence for entry
            adx_thresh: Minimum ADX for entry
            stop_atr: Stop loss ATR multiplier
            target_atr: Profit target ATR multiplier
            tick_size: Contract tick size for calculations
            profit_target: Profit target in dollars (default: 6000)
            max_loss_limit: Maximum loss limit in dollars (default: 3000)
            enable_trailing_stop: Enable trailing stops
        """
        # Initialize base class
        super().__init__(
            contract=contract,
            size=size,
            timeframe_minutes=timeframe_minutes,
            strategy=strategy,
            entry_conf=entry_conf,
            adx_thresh=adx_thresh,
            stop_pts=stop_pts,
            target_pts=target_pts,
            enable_trailing_stop=enable_trailing_stop            
        )
        
        # Simulation-specific attributes
        self.csv_path = csv_path
        self.tick_size = tick_size
        self.profit_target = profit_target
        self.max_loss_limit = max_loss_limit
        self.simulation_days = simulation_days
        
        # Pending entry system (sim bot only)
        self.pending_entry = None
        
        # Performance tracking
        self.total_pnl = 0.0
        self.total_pnl_dollars = 0.0
        self.trade_count = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.trades_log = []
        
        # Track current bar timestamp for display
        self.current_timestamp = None
        
        print(f"🤖 Simulation Bot initialized for {self.contract}")
        print(f"📈 Trade Params: Entry={self.entry_conf}, ADX={self.adx_thresh}, "
              f"Stop={self.stop_pts} pts, Target={self.target_pts} pts")
        print(f"📊 Strategy: {self.strategy.__class__.__name__}")
        print(f"💰 Profit Target: ${self.profit_target:,.2f} | Max Loss: ${self.max_loss_limit:,.2f}")

    def _load_csv_data(self):
        """
        Load historical data from CSV.
        
        Expected CSV format:
        time,open,high,low,close,volume
        
        Returns:
            DataFrame: Historical OHLCV data indexed by timestamp
        """
        try:
            df = pd.read_csv(self.csv_path)
            df.columns = df.columns.str.lower()
            
            logging.info(f"📊 CSV columns found: {list(df.columns)}")
            
            # Validate required OHLC columns
            required_cols = ['open', 'high', 'low', 'close']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                raise ValueError(f"CSV missing required columns: {missing_cols}")
            
            # Handle timestamp column (time, timestamp, or date)
            time_col = None
            if 'time' in df.columns:
                time_col = 'time'
            elif 'timestamp' in df.columns:
                time_col = 'timestamp'
            elif 'date' in df.columns:
                time_col = 'date'
            else:
                raise ValueError("CSV must have a 'time', 'timestamp', or 'date' column")
            
            # Handle volume column (optional)
            if 'volume' not in df.columns:
                logging.warning("⚠️ No volume column found - using 0 for all bars")
                df['volume'] = 0
            
            # Convert timestamp to datetime
            # Handle both Unix timestamps (numbers) and datetime strings
            if df[time_col].dtype in ['int64', 'float64']:
                # Unix timestamp - convert from seconds
                logging.info(f"📅 Detected Unix timestamp format in '{time_col}' column, converting...")
                df['timestamp'] = pd.to_datetime(df[time_col], unit='s')
            else:
                # String format - parse as datetime
                logging.info(f"📅 Parsing datetime strings from '{time_col}' column...")
                df['timestamp'] = pd.to_datetime(df[time_col])
            
            # Keep only necessary columns
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            logging.info(f"✅ Loaded {len(df)} bars from {self.csv_path}")
            logging.info(f"📅 Date range: {df.index[0]} to {df.index[-1]}")
            logging.info(f"📊 Price range: Low={df['low'].min():.2f}, High={df['high'].max():.2f}")
            
            return df
            
        except Exception as e:
            logging.exception(f"❌ Error loading CSV data: {e}")
            raise

    def _points_to_dollars(self, points):
        """
        Convert points to dollars based on contract specifications.
        For futures, typically: dollars = points * multiplier
        
        Args:
            points: P&L in points
            
        Returns:
            float: P&L in dollars
        """
        # Common multipliers: ES=50, NQ=20, RTY=50, YM=5
        # Get contract symbol to determine multiplier
        contract_symbol = CONTRACTS.get(self.contract, None)
        
        if contract_symbol == "NQ":
            multiplier = 20  # NQ = $20 per point
        elif contract_symbol == "ES":
            multiplier = 50  # ES = $50 per point
        elif contract_symbol == "RTY":
            multiplier = 50  # RTY = $50 per point
        elif contract_symbol == "YM":
            multiplier = 5   # YM = $5 per point
        else:
            multiplier = 50  # Default multiplier
            
        return points * multiplier

    async def _place_order(self, side, close_price, stop_loss, profit_target, stop_ticks, take_profit_ticks):
        """
        Store order data in pending_entry for execution on next bar.
        
        Args:
            side: 0 for LONG, 1 for SHORT
            close_price: Reference price (bar close where signal generated)
            stop_loss: Stop loss price
            profit_target: Profit target price
            stop_ticks: Stop loss in ticks
            take_profit_ticks: Profit target in ticks
        """
        direction = 'LONG' if side == 0 else 'SHORT'
        
        # Store for next bar entry
        self.pending_entry = {
            'direction': direction,
            'reference_price': close_price,
            'stop_loss': stop_loss,
            'profit_target': profit_target,
            'stop_ticks': stop_ticks,
            'take_profit_ticks': take_profit_ticks
        }
        
        # Log the pending order
        logging.debug(f"📝 Pending {direction} order: stop_ticks={stop_ticks}, "
                     f"take_profit_ticks={take_profit_ticks}")

    def _get_tick_size(self):
        """Get tick size for the contract."""
        return self.tick_size
    
    def _log_entry(self, side, entry_price, stop_loss, profit_target):
        """Override base class to show timestamp and more details."""
        print("\n" + "="*70)
        print(f"🚀 ENTRY {side}")
        print(f"   Time:   {self.current_timestamp}")
        print(f"   Price:  {entry_price:.2f}")
        print(f"   Stop:   {stop_loss:.2f} (Risk: {abs(entry_price - stop_loss):.2f} points)")
        print(f"   Target: {profit_target:.2f} (Reward: {abs(profit_target - entry_price):.2f} points)")
        
        # Calculate risk/reward ratio
        risk = abs(entry_price - stop_loss)
        reward = abs(profit_target - entry_price)
        rr_ratio = reward / risk if risk > 0 else 0
        print(f"   R:R:    1:{rr_ratio:.2f}")
        print("="*70 + "\n")
        
        logging.info(f"ENTRY {side} @ {entry_price:.2f} | Time: {self.current_timestamp} | "
                    f"Stop: {stop_loss:.2f} | Target: {profit_target:.2f}")
    
    def _log_exit(self, exit_price, exit_reason, pnl):
        """Override base class to show timestamp and P&L details."""
        pnl_dollars = self._points_to_dollars(pnl)
        
        print("\n" + "="*70)
        print(f"🛑 EXIT {self.position_type} - {exit_reason}")
        print(f"   Time:       {self.current_timestamp}")
        print(f"   Entry:      {self.entry_price:.2f}")
        print(f"   Exit:       {exit_price:.2f}")
        print(f"   P&L Points: {pnl:+.2f}")
        print(f"   P&L $:      ${pnl_dollars:+,.2f}")
        print(f"   Total P&L:  ${self.total_pnl_dollars:+,.2f}")
        print("="*70 + "\n")
        
        logging.info(f"EXIT {self.position_type} @ {exit_price:.2f} ({exit_reason}) | "
                    f"Time: {self.current_timestamp} | Entry: {self.entry_price:.2f} | "
                    f"P&L: {pnl:.2f} points (${pnl_dollars:,.2f})")

    async def _process_bar(self, bar_data, bar_index, total_bars):
        """
        Process a single bar from historical data.
        ✅ FIXED: Proper entry timing (enter at next bar open after signal)
        """
        try:
            timestamp, bar = bar_data
            self.current_timestamp = timestamp
            
            open_price = bar['open']
            high = bar['high']
            low = bar['low']
            close = bar['close']
            volume = bar['volume']
            
            # ========================================
            # STEP 1: Process pending entry from PREVIOUS bar
            # ========================================
            if self.pending_entry and not self.in_position:
                # Signal was generated on bar[i-1] using close price
                # Now entering at bar[i] OPEN (realistic timing)
                entry_price = open_price  # ✅ CORRECT - realistic entry
                
                direction = self.pending_entry['direction']
                
                # Adjust stop/target based on actual entry vs reference
                # (in case open gapped significantly from yesterday's close)
                price_diff = entry_price - self.pending_entry['reference_price']
                
                if direction == 'LONG':
                    stop_loss = self.pending_entry['stop_loss'] + price_diff
                    profit_target = self.pending_entry['profit_target'] + price_diff
                else:  # SHORT
                    stop_loss = self.pending_entry['stop_loss'] + price_diff
                    profit_target = self.pending_entry['profit_target'] + price_diff
                
                # Set position state
                self.in_position = True
                self.position_type = direction
                self.entry_price = entry_price
                self.stop_loss = stop_loss
                self.profit_target = profit_target
                self.entry_timestamp = timestamp
                
                self._log_entry(direction, entry_price, stop_loss, profit_target)
                                
                # Clear pending entry
                self.pending_entry = None
            
            # ========================================
            # STEP 2: Check exits if in position
            # ========================================
            if self.in_position:
                # Determine which price was hit first (simple heuristic)
                if abs(open_price - low) < abs(open_price - high):
                    prices_to_check = [low, high, close]
                else:
                    prices_to_check = [high, low, close]
                
                for price in prices_to_check:
                    exit_price, exit_reason = self._check_exit_conditions(price)
                    
                    if exit_reason:
                        pnl_points = self._calculate_pnl(exit_price)
                        pnl_dollars = self._points_to_dollars(pnl_points)
                        
                        self.total_pnl += pnl_points
                        self.total_pnl_dollars += pnl_dollars
                        self.trade_count += 1
                        
                        if pnl_points > 0:
                            self.winning_trades += 1
                        else:
                            self.losing_trades += 1
                        
                        # Log trade
                        trade_info = {
                            'entry_timestamp': self.entry_timestamp,
                            'exit_timestamp': timestamp,
                            'type': self.position_type,
                            'entry': self.entry_price,
                            'exit': exit_price,
                            'reason': exit_reason,
                            'pnl_points': pnl_points,
                            'pnl_dollars': pnl_dollars,
                            'total_pnl_dollars': self.total_pnl_dollars
                        }
                        self.trades_log.append(trade_info)
                        
                        self._log_exit(exit_price, exit_reason, pnl_points)
                        self._reset_position_state()
                        
                        # Check profit/loss limits
                        if self.profit_target is not None and self.total_pnl_dollars >= self.profit_target:
                            print("\n" + "="*50)
                            print(f"🎉 PROFIT TARGET REACHED: ${self.total_pnl_dollars:,.2f}")
                            print("="*50 + "\n")
                            return True
                        
                        if self.max_loss_limit is not None and self.total_pnl_dollars <= -self.max_loss_limit:
                            print("\n" + "="*50)
                            print(f"⛔ MAX LOSS LIMIT HIT: ${self.total_pnl_dollars:,.2f}")
                            print("="*50 + "\n")
                            return True
                        
                        break  # Exit processed
            
            # ========================================
            # STEP 3: Add CURRENT bar to history
            # ========================================
            bar_dict = {
                'timestamp': timestamp.isoformat(),
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            }
            self.historical_bars.append(bar_dict)
            
            # ========================================
            # STEP 4: Run strategy (generates signal for NEXT bar)
            # ========================================
            if len(self.historical_bars) >= self.num_historical_candles_needed:
                # Prediction using bar[i] close is now available
                # Entry will happen at bar[i+1] open!
                await self._run_ai_prediction()
            else:
                # Still collecting bars
                if bar_index % 1000 == 0:
                    logging.info(f"📊 Collecting bars: {len(self.historical_bars)}/{self.num_historical_candles_needed}")
            
            # Progress indicator
            if bar_index % 5000 == 0:
                progress = (bar_index / total_bars) * 100
                print(f"📊 Progress: {progress:.1f}% ({bar_index:,}/{total_bars:,} bars) | "
                    f"Trades: {self.trade_count} | P&L: ${self.total_pnl_dollars:+,.2f}")
                logging.info(f"📊 Progress: {progress:.1f}% | P&L: ${self.total_pnl_dollars:,.2f}")
            
            return False  # Continue simulation
            
        except Exception as e:
            logging.exception(f"❌ Error processing bar at index {bar_index}: {e}")
            return False

    def _print_summary(self):
        """Print simulation summary statistics."""
        print("\n" + "="*60)
        print("📊 SIMULATION RESULTS")
        print("="*60)
        print(f"Total Trades: {self.trade_count}")
        print(f"Winning Trades: {self.winning_trades}")
        print(f"Losing Trades: {self.losing_trades}")
        
        if self.trade_count > 0:
            win_rate = (self.winning_trades / self.trade_count) * 100
            print(f"Win Rate: {win_rate:.2f}%")
            
            # Calculate average P&L per trade
            avg_pnl = self.total_pnl_dollars / self.trade_count
            print(f"Average P&L per Trade: ${avg_pnl:,.2f}")
        
        print(f"\nTotal P&L (Points): {self.total_pnl:.2f}")
        print(f"Total P&L (Dollars): ${self.total_pnl_dollars:,.2f}")
                
        if self.profit_target is not None and self.total_pnl_dollars >= self.profit_target:
            print(f"🎉 Profit Target Hit: {self.total_pnl_dollars:,.2f} USD (Target: {self.profit_target:,.2f} USD)")
        
        # NEW CODE: Only perform comparison if self.max_loss_limit is set (not None)
        if self.max_loss_limit is not None and self.total_pnl_dollars <= -self.max_loss_limit:
            print(f"🛑 Max Loss Limit Hit: {self.total_pnl_dollars:,.2f} USD (Limit: -{self.max_loss_limit:,.2f} USD)")
        
        print("="*60 + "\n")
        
        # Print individual trades
        if self.trades_log:
            print("📝 TRADE LOG:")
            print("-"*60)
            for i, trade in enumerate(self.trades_log, 1):
                print(f"\nTrade #{i}:")
                print(f"  Time: {trade['timestamp']}")
                print(f"  Type: {trade['type']}")
                print(f"  Entry: {trade['entry']:.2f}")
                print(f"  Exit: {trade['exit']:.2f}")
                print(f"  Reason: {trade['reason']}")
                print(f"  P&L: {trade['pnl_points']:.2f} points (${trade['pnl_dollars']:,.2f})")
                print(f"  Cumulative P&L: ${trade['total_pnl_dollars']:,.2f}")

    async def run(self):
        """Run the simulation."""
        print(f"🚀 Starting simulation from {self.csv_path}...")
        
        # Load CSV data
        df = self._load_csv_data()

        if self.simulation_days is not None and self.simulation_days > 0:
            
            # Find the latest (most recent) date in the loaded data from the index
            latest_date = df.index.max()
            
            # Calculate the required start date by subtracting the specified number of days
            simulation_start_date = latest_date - pd.Timedelta(days=self.simulation_days)
            
            # Filter the DataFrame using the index
            original_len = len(df)
            # The filter now selects bars where the index (timestamp) is GREATER than or equal to the start date
            df = df[df.index >= simulation_start_date].copy() 

            if len(df) == 0:
                 print(f"🛑 Error: No data found within the last {self.simulation_days} days (starting from {simulation_start_date.strftime('%Y-%m-%d %H:%M:%S')}).")
                 return
            
            logging.info(f"📅 Limiting simulation to the last {self.simulation_days} days.")
            logging.info(f"Start date limit: {simulation_start_date.strftime('%Y-%m-%d %H:%M:%S')}")
            logging.info(f"End date: {latest_date.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Data reduced from {original_len} bars to {len(df)} bars.")
        
        # Get contract symbol from contract ID
        contract_symbol = CONTRACTS.get(self.contract)
        if not contract_symbol:
            logging.warning(f"⚠️ Unknown contract ID: {self.contract}, using as-is")
            contract_symbol = self.contract
        
        logging.info(f"📊 Contract: {self.contract} → Symbol: {contract_symbol}")
        logging.info(f"📊 Bars needed for strategy: {self.num_historical_candles_needed}")
        
        # Initialize strategy with contract symbol
        self.strategy.set_contract_symbol(contract_symbol)
        
        # Load strategy model and scaler
        self.strategy.load_model()
        self.strategy.load_scaler()
        
        print(f"📊 Processing {len(df)} bars...\n")
        logging.info(f"📊 Starting to process {len(df)} bars...")
        
        # Process each bar from the DataFrame
        # df.iterrows() returns (index, Series) tuples
        stop_simulation = False
        for idx, bar_data in enumerate(df.iterrows(), start=1):
            stop_simulation = await self._process_bar(bar_data, idx, len(df))
            if stop_simulation:
                logging.info(f"🛑 Simulation stopped at bar {idx}")
                break
        
        logging.info(f"✅ Finished processing. Total trades: {self.trade_count}")
        
        # Print summary
        self._print_summary()