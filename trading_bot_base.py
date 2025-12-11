#!/usr/bin/env python3
"""
Base Trading Bot Class - Simplified for live trading
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
        
        # State Management (for sim bot)
        self.in_position = False
        self.position_type = None
        self.entry_price = None
        self.stop_loss = None
        self.profit_target = None
        self.stop_orderId = None
        self.limit_orderId = None
        self.entry_timestamp = None
        
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
        self.entry_timestamp = None

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
        Calls _place_order() which has different implementations in live vs sim bots.
        """
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
                # ✅ CHECK FOR EXISTING POSITION BEFORE ENTERING
                has_position = await self._has_existing_position()
                if has_position:
                    print("⚠️ SIGNAL IGNORED - Already in position")
                    logging.info(f"{direction} signal ignored: Already in position")
                    return
                
                close_price = latest_bar['close']
                tick_size = self._get_tick_size()
                
                if direction == 'LONG':
                    # Calculate stop/target
                    stop_loss = close_price - self.stop_pts
                    profit_target = close_price + self.target_pts
                    stop_ticks = -int(self.stop_pts / tick_size)
                    take_profit_ticks = int(self.target_pts / tick_size)
                    
                    # Pass data to subclass implementation
                    await self._place_order(
                        side=0,
                        close_price=close_price,
                        stop_loss=stop_loss,
                        profit_target=profit_target,
                        stop_ticks=stop_ticks,
                        take_profit_ticks=take_profit_ticks
                    )
                    
                    print("="*40)
                    print(f"🔥 LONG SIGNAL")
                    print(f"  Reference: {close_price:.2f}")
                    print(f"  Stop: {stop_loss:.2f} | Target: {profit_target:.2f}")
                    print("="*40)
                    logging.info(f"LONG SIGNAL @ {close_price:.2f} Stop: {stop_loss:.2f} Target: {profit_target:.2f}")
                    
                else:  # SHORT
                    # Calculate stop/target
                    stop_loss = close_price + self.stop_pts
                    profit_target = close_price - self.target_pts
                    stop_ticks = int(self.stop_pts / tick_size)
                    take_profit_ticks = -int(self.target_pts / tick_size)
                    
                    # Pass data to subclass implementation
                    await self._place_order(
                        side=1,
                        close_price=close_price,
                        stop_loss=stop_loss,
                        profit_target=profit_target,
                        stop_ticks=stop_ticks,
                        take_profit_ticks=take_profit_ticks
                    )
                    
                    print("="*40)
                    print(f"🥶 SHORT SIGNAL")
                    print(f"  Reference: {close_price:.2f}")
                    print(f"  Stop: {stop_loss:.2f} | Target: {profit_target:.2f}")
                    print("="*40)
                    logging.info(f"SHORT SIGNAL @ {close_price:.2f} Stop: {stop_loss:.2f} Target: {profit_target:.2f}")
                    
        except Exception as e:
            logging.exception(f"❌ Error during AI prediction: {e}")

    @abstractmethod
    def _get_tick_size(self):
        """Get tick size for the contract. Must be implemented by subclass."""
        pass

    @abstractmethod
    async def _has_existing_position(self):
        """
        Check if there's an existing position. Must be implemented by subclass.
        
        For SimBot: Check internal state (self.in_position)
        For LiveBot: Call broker API to check actual positions
        
        Returns:
            bool: True if position exists, False otherwise
        """
        pass

    @abstractmethod
    async def _place_order(self, side, close_price, stop_loss, profit_target, stop_ticks, take_profit_ticks):
        """
        Place an order. Must be implemented by subclass.
        
        Args:
            side: 0 for LONG, 1 for SHORT
            close_price: Reference price (bar close)
            stop_loss: Stop loss price
            profit_target: Profit target price
            stop_ticks: Stop loss in ticks
            take_profit_ticks: Profit target in ticks
        """
        pass

    @abstractmethod
    async def run(self):
        """Main run loop. Must be implemented by subclass."""
        pass