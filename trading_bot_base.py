#!/usr/bin/env python3
"""
Base Trading Bot Class - FIXED for proper entry timing
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
        """Initialize the trading bot base."""
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
        
        # ✅ NEW: Pending entry system
        self.pending_entry = None  # Stores signal for next bar
        self.entry_timestamp = None  # Track when position entered
        
        # Strategy
        self.strategy = strategy
        
        # Historical bars
        seq_len = self.strategy.get_sequence_length()
        self.num_historical_candles_needed = 900 
        self.historical_bars = deque(maxlen=self.num_historical_candles_needed)
        
        # Trading parameters
        self.entry_conf = entry_conf
        self.adx_thresh = adx_thresh
        self.stop_pts = stop_pts 
        self.target_pts = target_pts
        
        logging.info(f"📊 Strategy: {self.strategy.__class__.__name__}")
        logging.info(f"📈 Trade Params: Entry={self.entry_conf}, ADX={self.adx_thresh}, "
                    f"Stop={self.stop_pts} pts, Target={self.target_pts} pts")

    def _check_exit_conditions(self, current_price):
        """Check if exit conditions are met for current position."""
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
        """Calculate PnL for current position."""
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
        self.entry_timestamp = None  # ✅ NEW

    def _log_exit(self, exit_price, exit_reason, pnl):
        """Log exit information."""
        print("="*40)
        print(f"🛑 EXIT {self.position_type} @ {exit_price:.2f} ({exit_reason})")
        print(f"  Entry: {self.entry_price:.2f} | PnL Points: {pnl:.2f}")
        print("="*40)
        logging.info(f"EXIT {self.position_type} @ {exit_price:.2f} ({exit_reason}) "
                    f"Entry: {self.entry_price:.2f} | PnL Points: {pnl:.2f}")

    def _log_entry(self, side, entry_price, stop_loss, profit_target):
        """Log entry information."""
        print("="*40)
        print(f"🚀 ENTRY {side} @ {entry_price:.2f}")
        print(f"  Stop: {stop_loss:.2f} | Target: {profit_target:.2f}")
        print("="*40)
        logging.info(f"ENTRY {side} @ {entry_price:.2f} Stop: {stop_loss:.2f} Target: {profit_target:.2f}")

    async def _run_ai_prediction(self):
        """
        Run AI prediction using the strategy.
        ✅ FIXED: Now sets pending_entry instead of entering immediately
        """
        if self.in_position or self.pending_entry:
            return
        
        try:
            # Convert historical bars to DataFrame
            df = pd.DataFrame(list(self.historical_bars))
            
            # Convert timestamp column to datetime and set as index
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            
            logging.debug(f"🔍 Running AI prediction with {len(df)} bars")
            
            # Add strategy-specific features
            df = self.strategy.add_features(df)
            
            # Check if warming up
            if df.empty:
                logging.warning("⚠️ Strategy is warming up (not enough data). Skipping prediction.")
                return
            
            # Validate features
            if not self.strategy.validate_features(df):
                logging.error("❌ Feature validation failed")
                return
            
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
            print(f"🤖 AI: {pred_labels[prediction]} (Conf: {confidence:.2%})")
            logging.info(f"AI: {pred_labels[prediction]} (Conf: {confidence:.2%}) ADX: {latest_bar.get('adx', 0):.1f}")
            
            if should_enter:
                close_price = latest_bar['close']
                atr = latest_bar.get('atr', 0)
                                                
                tick_size = self._get_tick_size()
                
                # ✅ FIXED: Set pending_entry instead of entering immediately
                if direction == 'LONG':
                    # Calculate stop/target based on close_price
                    stop_loss = close_price - self.stop_pts
                    profit_target = close_price + self.target_pts
                    
                    # Store for next bar
                    self.pending_entry = {
                        'direction': 'LONG',
                        'reference_price': close_price,  # For stop/target calculation
                        'stop_loss': stop_loss,
                        'profit_target': profit_target,
                        'stop_ticks': -int(self.stop_pts / tick_size),
                        'take_profit_ticks': int(self.target_pts / tick_size)
                    }
                    
                    print("="*40)
                    print(f"🔥 SIGNAL: LONG (will enter next bar open)")
                    print(f"  Reference: {close_price:.2f}")
                    print(f"  Planned SL: {stop_loss:.2f} | PT: {profit_target:.2f}")
                    print("="*40)
                    logging.info(f"SIGNAL LONG @ {close_price:.2f} (pending next bar)")
                    
                else:  # SHORT
                    # Calculate stop/target based on close_price
                    stop_loss = close_price + self.stop_pts
                    profit_target = close_price - self.target_pts
                    
                    # Store for next bar
                    self.pending_entry = {
                        'direction': 'SHORT',
                        'reference_price': close_price,
                        'stop_loss': stop_loss,
                        'profit_target': profit_target,
                        'stop_ticks': int(self.stop_pts / tick_size),
                        'take_profit_ticks': -int(self.target_pts / tick_size)
                    }
                    
                    print("="*40)
                    print(f"🥶 SIGNAL: SHORT (will enter next bar open)")
                    print(f"  Reference: {close_price:.2f}")
                    print(f"  Planned SL: {stop_loss:.2f} | PT: {profit_target:.2f}")
                    print("="*40)
                    logging.info(f"SIGNAL SHORT @ {close_price:.2f} (pending next bar)")
                    
        except Exception as e:
            logging.exception(f"❌ Error during AI prediction: {e}")

    @abstractmethod
    def _get_tick_size(self):
        """Get tick size for the contract. Must be implemented by subclass."""
        pass

    @abstractmethod
    async def _place_order(self, side, type=2, stop_ticks=10, take_profit_ticks=20):
        """Place an order. Must be implemented by subclass."""
        pass

    @abstractmethod
    async def run(self):
        """Main run loop. Must be implemented by subclass."""
        pass