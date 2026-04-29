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
from strategies.strategy_base import BaseStrategy
from bots.trading_bot_base import TradingBot
from utils.bot_utils import TICK_VALUES, TICK_SIZES

CONTRACTS = {
    "CON.F.US.ENQ.Z25": "NQ",
    "CON.F.US.EP.Z25": "ES",
}


class SimulationBot(TradingBot):
    """Simulation bot for backtesting strategies"""
    _is_simulation = True

    def __init__(
        self,
        csv_path,
        contract,
        size,
        timeframe_minutes,
        strategy: BaseStrategy,
        entry_conf,
        stop_pts,
        target_pts,
        tick_size=0.01,
        profit_target=6000,
        max_loss_limit=3000,
        enable_trailing_stop=False,
        simulation_days=None,
        start_date=None,
        end_date=None,
        risk_amount=None,
        high_conf_multiplier=1.0,
        max_contracts=15,
        min_stop_pts=1.0,
        min_stop_atr_mult=0.5,
        breakeven_on_2r=False,
        df=None,
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
            stop_pts=stop_pts,
            target_pts=target_pts,
            enable_trailing_stop=enable_trailing_stop,
            risk_amount=risk_amount,
            high_conf_multiplier=high_conf_multiplier,
            max_contracts=max_contracts,
            min_stop_pts=min_stop_pts,
            min_stop_atr_mult=min_stop_atr_mult,
            breakeven_on_2r=breakeven_on_2r,
        )

        # Simulation-specific attributes
        self.csv_path = csv_path
        self._preloaded_df = df  # pre-fetched DataFrame (live-data mode); takes priority over csv_path
        self.tick_size = tick_size
        self.session_profit_target = profit_target  # dollar P&L target (not trade take-profit price)
        self.initial_max_loss = max_loss_limit       # never changes
        self.mll = max_loss_limit                    # grows EOD when PnL >= MLL
        self.max_loss_limit = max_loss_limit         # kept for summary/display compat
        self.simulation_days = simulation_days
        self.start_date = start_date
        self.end_date = end_date
        self._last_bar_date = None                   # for EOD detection
        
        # Pending entry system (sim bot only)
        self.pending_entry = None
        self._current_signal_meta = {}
        self.current_trade_size = size
        
        # Track if we just entered this bar (prevents same-bar exit)
        self.just_entered_this_bar = False
        
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
        print(f"📈 Trade Params: Entry={self.entry_conf}, "
              f"Stop={self.stop_pts} pts, Target={self.target_pts} pts")
        print(f"📊 Strategy: {self.strategy.__class__.__name__}")
        pt_str = f"${self.session_profit_target:,.2f}" if self.session_profit_target is not None else "none"
        print(f"💰 Profit Target: {pt_str} | Max Loss: ${self.max_loss_limit:,.2f}")

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
            
            # Handle timestamp column (time, timestamp, date, or datetime)
            time_col = None
            for candidate in ('time', 'timestamp', 'date', 'datetime'):
                if candidate in df.columns:
                    time_col = candidate
                    break
            if time_col is None:
                raise ValueError("CSV must have a 'time', 'timestamp', 'date', or 'datetime' column")
            
            # Handle volume column (optional)
            if 'volume' not in df.columns:
                logging.warning("⚠️ No volume column found - using 0 for all bars")
                df['volume'] = 0
            
            # Convert timestamp to datetime then localize to US/Eastern (matching training)
            if df[time_col].dtype in ['int64', 'float64']:
                logging.info(f"📅 Detected Unix timestamp format in '{time_col}' column, converting...")
                df['timestamp'] = pd.to_datetime(df[time_col], unit='s', utc=True)
            else:
                logging.info(f"📅 Parsing datetime strings from '{time_col}' column...")
                df['timestamp'] = pd.to_datetime(df[time_col], utc=True)
            df['timestamp'] = df['timestamp'].dt.tz_convert('America/New_York')
            logging.info("🕐 Timestamps converted to America/New_York (ET)")
            
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
        """Convert points to dollars using tick value / tick size ratio, scaled by position size."""
        tick_value = self._get_tick_value()
        tick_size = self._get_tick_size()
        point_value = tick_value / tick_size if tick_size != 0 else 1.0
        return points * point_value * self.current_trade_size

    async def _place_order(self, side, close_price, stop_loss, profit_target, stop_ticks, take_profit_ticks, size):
        """
        Store order data in pending_entry for execution on next bar.

        Args:
            side: 0 for LONG, 1 for SHORT
            close_price: Reference price (bar close where signal generated)
            stop_loss: Stop loss price
            profit_target: Profit target price
            stop_ticks: Stop loss in ticks
            take_profit_ticks: Profit target in ticks
            size: Number of contracts
        """
        direction = 'LONG' if side == 0 else 'SHORT'

        # Snapshot signal diagnostics from strategy at the moment of signal
        signal_meta = {}
        if hasattr(self.strategy, '_latest_signal_meta'):
            signal_meta = dict(self.strategy._latest_signal_meta)

        # Store for next bar entry
        self.pending_entry = {
            'direction':         direction,
            'reference_price':   close_price,
            'stop_loss':         stop_loss,
            'profit_target':     profit_target,
            'stop_ticks':        stop_ticks,
            'take_profit_ticks': take_profit_ticks,
            'size':              size,
            'signal_meta':       signal_meta,
        }
        
        # Log the pending order
        logging.debug(f"📝 Pending {direction} order: stop_ticks={stop_ticks}, "
                     f"take_profit_ticks={take_profit_ticks}")

    def _get_tick_size(self):
        """Get tick size from TICK_SIZES lookup, falling back to --tick_size arg."""
        parts = self.contract.upper().split('.')
        if len(parts) >= 4:
            symbol = parts[3]
            if symbol in TICK_SIZES:
                return TICK_SIZES[symbol]
        if self.contract.upper() in TICK_SIZES:
            return TICK_SIZES[self.contract.upper()]
        return self.tick_size

    def _get_tick_value(self):
        """Get dollar value per tick using hardcoded lookup (API not available in sim)."""
        parts = self.contract.upper().split('.')
        # Full contract ID format: CON.F.US.{SYMBOL}.{EXPIRY} — segment index 3 is the product code
        if len(parts) >= 4:
            symbol = parts[3]
            if symbol in TICK_VALUES:
                return TICK_VALUES[symbol]
        # Simple symbol format: "MNQ", "NQ", etc.
        if self.contract.upper() in TICK_VALUES:
            return TICK_VALUES[self.contract.upper()]
        return 0.50
    
    async def _has_existing_position(self):
        """
        Check if there's an existing position using internal state.
        
        Returns:
            bool: True if position exists, False otherwise
        """
        return self.in_position
    
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
        target_dist = abs(self.profit_target - self.entry_price) if (self.profit_target and self.entry_price) else 0
        mfe_pct = f" ({self.mfe_pts / target_dist * 100:.0f}% of target)" if target_dist else ""

        print("\n" + "="*70)
        print(f"🛑 EXIT {self.position_type} - {exit_reason}")
        print(f"   Time:       {self.current_timestamp}")
        print(f"   Entry:      {self.entry_price:.2f}")
        print(f"   Exit:       {exit_price:.2f}")
        print(f"   MFE:        +{self.mfe_pts:.2f}pts{mfe_pct}")
        print(f"   P&L Points: {pnl:+.2f}")
        print(f"   P&L $:      ${pnl_dollars:+,.2f}")
        print(f"   Total P&L:  ${self.total_pnl_dollars:+,.2f}")
        if self.mll is not None:
            print(f"   MLL: ${self.mll:,.2f}")
        print("="*70 + "\n")

        logging.info(f"EXIT {self.position_type} @ {exit_price:.2f} ({exit_reason}) | "
                    f"Time: {self.current_timestamp} | Entry: {self.entry_price:.2f} | "
                    f"MFE: +{self.mfe_pts:.2f}pts{mfe_pct} | "
                    f"P&L: {pnl:.2f} points (${pnl_dollars:,.2f})")

    async def _process_bar(self, bar_data, bar_index, total_bars):
        """
        Process a single bar from historical data.
        ✅ FIXED: Proper entry timing (enter at next bar open after signal)
        ✅ FIXED: Prevent same-bar entry/exit
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
            # EOD MLL RATCHET: on first bar of a new day, lock in MLL if PnL exceeded it
            # ========================================
            bar_date = timestamp.date()
            if self._last_bar_date is not None and bar_date != self._last_bar_date:
                if self.mll is not None and self.total_pnl_dollars >= self.mll:
                    old_mll = self.mll
                    self.mll = self.total_pnl_dollars
                    logging.info(f"📈 EOD MLL ratchet: ${old_mll:,.2f} → ${self.mll:,.2f}")
            self._last_bar_date = bar_date

            # ========================================
            # STEP 1: Process pending entry from PREVIOUS bar
            # ========================================
            if self.pending_entry and self.in_position:
                self.pending_entry = None  # discard stale signal — already in position

            if self.pending_entry and not self.in_position:
                # Signal was generated on bar[i-1] using close price
                # Now entering at bar[i] OPEN (realistic timing)
                entry_price = open_price

                direction = self.pending_entry['direction']

                # Adjust stop/target based on actual entry vs reference
                # (in case open gapped significantly from yesterday's close)
                price_diff = entry_price - self.pending_entry['reference_price']

                # Cancel entry if the gap exceeds the original stop distance —
                # price has moved so far that the zone is no longer valid.
                stored_stop_pts = abs(self.pending_entry['stop_loss'] - self.pending_entry['reference_price'])
                if stored_stop_pts > 0 and abs(price_diff) > stored_stop_pts:
                    logging.info(
                        f"⚠️ Pending {direction} entry cancelled — gap {price_diff:+.2f}pts "
                        f"exceeds stop {stored_stop_pts:.2f}pts (zone no longer valid)"
                    )
                    self.pending_entry = None

            if self.pending_entry and not self.in_position:
                entry_price = open_price
                direction = self.pending_entry['direction']
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
                self.current_trade_size = self.pending_entry.get('size', self.size)
                self._current_signal_meta = self.pending_entry.get('signal_meta', {})

                # Mark that we just entered this bar (prevents same-bar exit)
                self.just_entered_this_bar = True

                self._log_entry(direction, entry_price, stop_loss, profit_target)

                # Clear pending entry
                self.pending_entry = None
            
            # ========================================
            # STEP 2: Check exits if in position
            # Skip exit check if we just entered this bar
            # ========================================
            if self.in_position and not self.just_entered_this_bar:
                favorable_price = high if self.position_type == 'LONG' else low
                self._update_mfe(favorable_price)

                # ── Gap-open check: if bar opens past stop/target, fill at open ──
                exit_price, exit_reason = None, None
                if self.position_type == 'LONG':
                    if open_price <= self.stop_loss:
                        exit_price, exit_reason = open_price, 'STOP_LOSS'
                    elif open_price >= self.profit_target:
                        # TP is a limit order — gap fills at the TP level, not the open
                        exit_price, exit_reason = self.profit_target, 'PROFIT_TARGET'
                elif self.position_type == 'SHORT':
                    if open_price >= self.stop_loss:
                        exit_price, exit_reason = open_price, 'STOP_LOSS'
                    elif open_price <= self.profit_target:
                        # TP is a limit order — gap fills at the TP level, not the open
                        exit_price, exit_reason = self.profit_target, 'PROFIT_TARGET'

                # ── Intrabar check: wick or close touches stop or target ──
                if not exit_reason:
                    if self.position_type == 'LONG':
                        stop_hit        = low  <= self.stop_loss
                        wick_target_hit = high >= self.profit_target
                        close_target_hit = close >= self.profit_target
                        target_hit      = wick_target_hit or close_target_hit
                        if stop_hit and target_hit:
                            # Both levels hit intrabar. TP is a pre-placed limit order that
                            # fills the instant price touches it; stop can only fire after TP
                            # misses. Prefer TP when the wick reached it; fall back to the
                            # open-proximity heuristic only when the close crossed TP (no wick).
                            if wick_target_hit:
                                exit_price, exit_reason = self.profit_target, 'PROFIT_TARGET'
                            elif abs(open_price - low) <= abs(open_price - high):
                                exit_price, exit_reason = self.stop_loss,     'STOP_LOSS'
                            else:
                                exit_price, exit_reason = close,              'PROFIT_TARGET'
                        elif stop_hit:
                            exit_price, exit_reason = self.stop_loss,     'STOP_LOSS'
                        elif target_hit:
                            exit_price = self.profit_target if wick_target_hit else close
                            exit_reason = 'PROFIT_TARGET'
                    elif self.position_type == 'SHORT':
                        stop_hit        = high >= self.stop_loss
                        wick_target_hit = low  <= self.profit_target
                        close_target_hit = close <= self.profit_target
                        target_hit      = wick_target_hit or close_target_hit
                        if stop_hit and target_hit:
                            if wick_target_hit:
                                exit_price, exit_reason = self.profit_target, 'PROFIT_TARGET'
                            elif abs(open_price - high) <= abs(open_price - low):
                                exit_price, exit_reason = self.stop_loss,     'STOP_LOSS'
                            else:
                                exit_price, exit_reason = close,              'PROFIT_TARGET'
                        elif stop_hit:
                            exit_price, exit_reason = self.stop_loss,     'STOP_LOSS'
                        elif target_hit:
                            exit_price = self.profit_target if wick_target_hit else close
                            exit_reason = 'PROFIT_TARGET'

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

                    trade_info = {
                        'entry_timestamp':  self.entry_timestamp,
                        'exit_timestamp':   timestamp,
                        'type':             self.position_type,
                        'entry':            self.entry_price,
                        'exit':             exit_price,
                        'reason':           exit_reason,
                        'pnl_points':       pnl_points,
                        'pnl_dollars':      pnl_dollars,
                        'mfe_pts':          self.mfe_pts,
                        'total_pnl_dollars': self.total_pnl_dollars,
                        'signal_meta':      dict(self._current_signal_meta),
                    }
                    self.trades_log.append(trade_info)

                    self._log_exit(exit_price, exit_reason, pnl_points)
                    self.strategy.on_trade_exit(exit_reason)
                    self._reset_position_state()

                    if self.mll is not None and self.total_pnl_dollars <= -self.mll:
                        print("\n" + "="*50)
                        print(f"⛔ MLL HIT ZERO: P&L ${self.total_pnl_dollars:,.2f} "
                              f"(MLL: ${self.mll:,.2f})")
                        print("="*50 + "\n")
                        return True

                    if self.session_profit_target is not None and self.total_pnl_dollars >= self.session_profit_target:
                        print("\n" + "="*50)
                        print(f"🎉 PROFIT TARGET REACHED: ${self.total_pnl_dollars:,.2f}")
                        print("="*50 + "\n")
                        return True
                else:
                    # No exit this bar — update breakeven for future bars only.
                    # Must run after exit checks so the modified stop_loss cannot
                    # interfere with the current bar's gap-open or intrabar logic.
                    self._check_and_set_breakeven(favorable_price)
            
            # Reset the just_entered flag at end of bar processing
            self.just_entered_this_bar = False
            
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
                
        if self.session_profit_target is not None and self.total_pnl_dollars >= self.session_profit_target:
            print(f"🎉 Profit Target Hit: {self.total_pnl_dollars:,.2f} USD (Target: {self.session_profit_target:,.2f} USD)")
        
        if self.mll is not None and self.total_pnl_dollars <= -self.mll:
            print(f"🛑 MLL Hit Zero: P&L ${self.total_pnl_dollars:,.2f} (MLL: ${self.mll:,.2f})")

        # Aggregate skip reasons from bot + strategy
        strat_skips = getattr(self.strategy, 'skip_stats', {})
        bot_skips   = self.skip_stats
        skip_rows = [
            ("Conf gate",      strat_skips.get('conf_gate', 0)),
            ("RR gate",        strat_skips.get('rr_gate', 0)),
            ("Hold (no zone)", strat_skips.get('hold', 0)),
            ("Stop too wide",  bot_skips.get('stop_too_wide', 0)),
            ("Stop too tight", bot_skips.get('stop_too_tight', 0)),
            ("Predict error",  bot_skips.get('predict_error', 0)),
        ]
        total_skips = sum(n for _, n in skip_rows)
        if total_skips > 0:
            print("\n--- Signal Filter Summary ---")
            for label, count in skip_rows:
                if count > 0:
                    print(f"  {label:<18} {count:>5} signals skipped")
            print(f"  {'Total skipped':<18} {total_skips:>5}")

        print("="*60 + "\n")
        
        # Signal feature analysis: winners vs losers
        trades_with_meta = [t for t in self.trades_log if t.get('signal_meta')]
        if trades_with_meta:
            winners = [t for t in trades_with_meta if t['pnl_points'] > 0]
            losers  = [t for t in trades_with_meta if t['pnl_points'] <= 0]
            meta_keys = list(trades_with_meta[0]['signal_meta'].keys())

            def avg(trades, key):
                vals = [t['signal_meta'].get(key) for t in trades]
                vals = [v for v in vals if isinstance(v, (int, float))]
                return sum(vals) / len(vals) if vals else 0.0

            print("\n📊 SIGNAL FEATURE ANALYSIS (winners vs losers):")
            print("-"*60)
            header = f"  {'Feature':<22} {'Winners':>10} {'Losers':>10} {'Delta':>10}"
            print(header)
            print("  " + "-"*56)
            for key in meta_keys:
                w = avg(winners, key)
                l = avg(losers,  key)
                print(f"  {key:<22} {w:>10.4f} {l:>10.4f} {w-l:>+10.4f}")
            print(f"  {'trade_count':<22} {len(winners):>10} {len(losers):>10}")
            print("-"*60)

        # Print individual trades
        if self.trades_log:
            print("\n📋 TRADE LOG:")
            print("-"*60)
            for i, trade in enumerate(self.trades_log, 1):
                print(f"\nTrade #{i}:")
                print(f"  Entry Time: {trade['entry_timestamp']}")
                print(f"  Exit Time: {trade['exit_timestamp']}")
                print(f"  Type: {trade['type']}")
                print(f"  Entry: {trade['entry']:.2f}")
                print(f"  Exit: {trade['exit']:.2f}")
                print(f"  Reason: {trade['reason']}")
                print(f"  P&L: {trade['pnl_points']:.2f} points (${trade['pnl_dollars']:,.2f})")
                print(f"  Cumulative P&L: ${trade['total_pnl_dollars']:,.2f}")
                if trade.get('signal_meta'):
                    m = trade['signal_meta']
                    print(f"  Signal: prob={m.get('signal_prob','?')} rr={m.get('risk_rr','?')} "
                          f"confluence={m.get('confluence_score','?')} "
                          f"htf={m.get('htf_trend','?')} aligned={m.get('trend_alignment','?')}")

    async def run(self):
        """Run the simulation."""
        if self._preloaded_df is not None:
            print(f"🚀 Starting simulation from live API data...")
            df = self._preloaded_df.copy()
        else:
            print(f"🚀 Starting simulation from {self.csv_path}...")
            df = self._load_csv_data()

        original_len = len(df)
        if self.start_date is not None or self.end_date is not None:
            start = pd.Timestamp(self.start_date, tz=df.index.tz) if self.start_date else df.index.min()
            end = pd.Timestamp(self.end_date, tz=df.index.tz) + pd.Timedelta(days=1) if self.end_date else df.index.max() + pd.Timedelta(seconds=1)
            df = df[(df.index >= start) & (df.index < end)].copy()
            if len(df) == 0:
                print(f"🛑 Error: No data found between {self.start_date} and {self.end_date}.")
                return
            logging.info(f"📅 Date range filter: {self.start_date} → {self.end_date}")
            print(f"Data reduced from {original_len} bars to {len(df)} bars.")
        elif self.simulation_days is not None and self.simulation_days > 0:
            latest_date = df.index.max()
            simulation_start_date = latest_date - pd.Timedelta(days=self.simulation_days)
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
        
        self.strategy.load_model()
        
        print(f"📊 Processing {len(df)} bars...\n")
        logging.info(f"📊 Starting to process {len(df)} bars...")
        
        # Process each bar from the DataFrame
        stop_simulation = False
        for idx, bar_data in enumerate(df.iterrows(), start=1):
            stop_simulation = await self._process_bar(bar_data, idx, len(df))
            if stop_simulation:
                logging.info(f"🛑 Simulation stopped at bar {idx}")
                break
        
        logging.info(f"✅ Finished processing. Total trades: {self.trade_count}")
        
        # Print summary
        self._print_summary()