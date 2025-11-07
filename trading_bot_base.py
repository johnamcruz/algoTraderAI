#!/usr/bin/env python3
"""
Base Trading Bot Class

This abstract base class contains common functionality shared between
RealTimeBot and SimulationBot. It handles:
1. Strategy management
2. Position state tracking
3. Entry/exit logic
4. Bar management
"""

import logging
import pandas as pd
from collections import deque
from datetime import datetime, timezone
from abc import ABC, abstractmethod
from strategy_base import BaseStrategy


class TradingBot(ABC):
    """Abstract base class for trading bots"""
    
    def __init__(
        self,
        contract,
        size,
        timeframe_minutes,
        strategy: BaseStrategy,
        entry_conf,
        adx_thresh,
        stop_pts,
        target_pts,
        enable_trailing_stop=False
    ):
        """
        Initialize the trading bot base.
        
        Args:
            contract: Contract ID
            size: Position size
            timeframe_minutes: Bar timeframe
            strategy: Strategy instance (implements BaseStrategy)
            entry_conf: Minimum confidence for entry
            adx_thresh: Minimum ADX for entry
            stop_atr: Stop loss in points
            target_atr: Profit target in points
            enable_trailing_stop: Enable trailing stops
        """
        self.contract = contract
        self.size = size
        self.timeframe_minutes = int(timeframe_minutes)
        self.enable_trailing_stop = enable_trailing_stop
        
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
        
        # Historical bars (strategy determines how many needed)
        seq_len = self.strategy.get_sequence_length()
        # --- MODIFICATION: Need more history for warm-up than just seq_len ---
        # We need at least 200 bars for ema200 + seq_len for the model.
        # Let's set a safe buffer. 500 should be enough for most indicators.
        self.num_historical_candles_needed = 900 
        self.historical_bars = deque(maxlen=self.num_historical_candles_needed)
        # --- END MODIFICATION ---
        
        # Trading parameters
        self.entry_conf = entry_conf
        self.adx_thresh = adx_thresh
        self.stop_pts = stop_pts 
        self.target_pts = target_pts
        
        logging.info(f"📊 Strategy: {self.strategy.__class__.__name__}")
        logging.info(f"📈 Trade Params: Entry={self.entry_conf}, ADX={self.adx_thresh}, "
                    f"Stop={self.stop_pts} pts, Target={self.target_pts} pts")

    def _check_exit_conditions(self, current_price):
        """
        Check if exit conditions are met for current position.
        
        Args:
            current_price: Current market price
            
        Returns:
            tuple: (exit_price, exit_reason) or (None, None) if no exit
        """
        if not self.in_position:
            return None, None
            
        if self.stop_loss is None or self.profit_target is None:
            logging.error("⚠️ Exit check skipped: stop_loss or profit_target not set.")
            return None, None
            
        exit_price, exit_reason = None, None
        
        if self.position_type == 'LONG':
            if current_price <= self.stop_loss:
                exit_price, exit_reason = self.stop_loss, 'STOP_LOSS'
            elif current_price >= self.profit_target:
                exit_price, exit_reason = self.profit_target, 'PROFIT_TARGET'
        elif self.position_type == 'SHORT':
            if current_price >= self.stop_loss:
                exit_price, exit_reason = self.stop_loss, 'STOP_LOSS'
            elif current_price <= self.profit_target:
                exit_price, exit_reason = self.profit_target, 'PROFIT_TARGET'
                
        return exit_price, exit_reason

    def _calculate_pnl(self, exit_price):
        """
        Calculate PnL for current position.
        
        Args:
            exit_price: Exit price
            
        Returns:
            float: PnL in points
        """
        if self.position_type == 'LONG':
            return exit_price - self.entry_price
        elif self.position_type == 'SHORT':
            return self.entry_price - exit_price
        return 0.0

    def _reset_position_state(self):
        """Reset position state after exit."""
        self.in_position = False
        self.position_type = None
        self.entry_price = None
        self.stop_loss = None
        self.profit_target = None
        self.stop_orderId = None
        self.limit_orderId = None

    def _log_exit(self, exit_price, exit_reason, pnl):
        """
        Log exit information.
        
        Args:
            exit_price: Exit price
            exit_reason: Reason for exit
            pnl: Profit/loss in points
        """
        print("="*40)
        print(f"🛑 EXIT {self.position_type} @ {exit_price:.2f} ({exit_reason})")
        print(f"  Entry: {self.entry_price:.2f} | PnL Points: {pnl:.2f}")
        print("="*40)
        logging.info(f"EXIT {self.position_type} @ {exit_price:.2f} ({exit_reason}) "
                    f"Entry: {self.entry_price:.2f} | PnL Points: {pnl:.2f}")

    def _log_entry(self, side, entry_price, stop_loss, profit_target):
        """
        Log entry information.
        
        Args:
            side: 'LONG' or 'SHORT'
            entry_price: Entry price
            stop_loss: Stop loss price
            profit_target: Profit target price
        """
        print("="*40)
        print(f"🚀 ENTRY {side} @ {entry_price:.2f}")
        print(f"  Stop: {stop_loss:.2f} | Target: {profit_target:.2f}")
        print("="*40)
        logging.info(f"ENTRY {side} @ {entry_price:.2f} Stop: {stop_loss:.2f} Target: {profit_target:.2f}")

    async def _run_ai_prediction(self):
        """
        Run AI prediction using the strategy.
        This is the common method used by both RealTimeBot and SimulationBot.
        """
        if self.in_position:
            return
        
        try:
            # Convert historical bars to DataFrame
            df = pd.DataFrame(list(self.historical_bars))
            
            # Convert timestamp column to datetime and set as index
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            
            logging.debug(f"🔍 Running AI prediction with {len(df)} bars")
            logging.debug(f"🔍 DataFrame columns: {list(df.columns)}")
            logging.debug(f"🔍 DataFrame index: {df.index.name}")
            
            # Add strategy-specific features
            df = self.strategy.add_features(df)
            
            # --- FIX: CATCH EMPTY DATAFRAME ---
            # This happens if add_features() drops all rows during warm-up
            if df.empty:
                logging.warning("⚠️ Strategy is warming up (not enough data). Skipping prediction.")
                return
            # --- END FIX ---
            
            logging.debug(f"🔍 After add_features, columns: {list(df.columns)}")
            
            # Validate features
            if not self.strategy.validate_features(df):
                logging.error("❌ Feature validation failed")
                return
            
            logging.debug(f"✅ Feature validation passed")
            
            # Get prediction from strategy
            prediction, confidence = self.strategy.predict(df)
            
            logging.debug(f"🔍 Prediction: {prediction}, Confidence: {confidence:.2%}")
            
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
                close_price = latest_bar['close']
                atr = latest_bar.get('atr', 0)
                
                if atr <= 0:
                    logging.error("❌ Invalid ATR, skipping entry")
                    return
                
                tick_size = self._get_tick_size()
                
                if direction == 'LONG':
                    self.in_position = True
                    self.position_type = 'LONG'
                    self.entry_price = close_price
                    # LONG SL/PT: Use fixed point distance
                    self.stop_loss = self.entry_price - self.stop_pts
                    self.profit_target = self.entry_price + self.target_pts
                    
                    print("="*40)
                    print(f"🔥🔥🔥 ENTERING LONG @ {self.entry_price:.2f} 🔥🔥🔥")
                    print(f"  SL: {self.stop_loss:.2f} | PT: {self.profit_target:.2f}")
                    print("="*40)
                    logging.info(f"LONG @ {self.entry_price:.2f} SL: {self.stop_loss:.2f} | PT: {self.profit_target:.2f}")
                    
                    # Calculate ticks: Use point distance directly
                    stop_loss_ticks = -int(self.stop_pts / tick_size)
                    take_profit_ticks = int(self.target_pts / tick_size)
                    
                    # Place order
                    await self._place_order(0, stop_ticks=stop_loss_ticks, take_profit_ticks=take_profit_ticks)
                    
                else:  # SHORT
                    self.in_position = True
                    self.position_type = 'SHORT'
                    self.entry_price = close_price
                    # SHORT SL/PT: Use fixed point distance
                    self.stop_loss = self.entry_price + self.stop_pts
                    self.profit_target = self.entry_price - self.target_pts
                    
                    print("="*40)
                    print(f"🥶🥶🥶 ENTERING SHORT @ {self.entry_price:.2f} 🥶🥶🥶")
                    print(f"  SL: {self.stop_loss:.2f} | PT: {self.profit_target:.2f}")
                    print("="*40)
                    logging.info(f"SHORT @ {self.entry_price:.2f} SL: {self.stop_loss:.2f} | PT: {self.profit_target:.2f}")
                    
                    # Calculate ticks: Use point distance directly
                    stop_loss_ticks = int(self.stop_pts / tick_size)
                    take_profit_ticks = -int(self.target_pts / tick_size)
                    
                    # Place order
                    await self._place_order(1, stop_ticks=stop_loss_ticks, take_profit_ticks=take_profit_ticks)
                    
        except Exception as e:
            logging.exception(f"❌ Error during AI prediction: {e}")

    @abstractmethod
    def _get_tick_size(self):
        """
        Get tick size for the contract. Must be implemented by subclass.
        
        Returns:
            float: Tick size
        """
        pass

    @abstractmethod
    async def _place_order(self, side, type=2, stop_ticks=10, take_profit_ticks=20):
        """
        Place an order. Must be implemented by subclass.
        
        Args:
            side: 1 for long, 0 for short
            type: Order type
            stop_ticks: Stop loss in ticks
            take_profit_ticks: Profit target in ticks
        """
        pass

    @abstractmethod
    async def run(self):
        """Main run loop. Must be implemented by subclass."""
        pass