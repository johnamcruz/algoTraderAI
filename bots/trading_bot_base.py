#!/usr/bin/env python3
"""
Base Trading Bot Class - Simplified for live trading
"""

import math
import logging
import pandas as pd
from collections import deque
from datetime import datetime, timezone
from abc import ABC, abstractmethod
from strategies.strategy_base import BaseStrategy


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
        stop_pts=None,
        target_pts=None,
        enable_trailing_stop=False,
        breakeven_on_2r=False,
        risk_amount=None,
        high_conf_multiplier=1.0,
        max_contracts=15,
        min_stop_pts=1.0,
        min_stop_atr_mult=0.5,
    ):
        """Initialize the trading bot base."""
        self.contract = contract
        self.size = size
        self.risk_amount = risk_amount
        self.high_conf_multiplier = high_conf_multiplier
        self.max_contracts = max_contracts
        self.min_stop_pts = min_stop_pts
        self.breakeven_on_2r = breakeven_on_2r
        self.min_stop_atr_mult = min_stop_atr_mult
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
        self.stop_bracket_order_id = None
        self.position_size = None
        self.entry_timestamp = None
        self.mfe_pts: float = 0.0       # peak unrealized gain in points this trade
        self.breakeven_set: bool = False    # True once stop has been moved to entry
        self._pre_breakeven_stop: float = 0.0  # original stop saved before BE fires (for rollback)

        # Strategy
        self.strategy = strategy
        
        # Historical bars
        self.num_historical_candles_needed = self.strategy.get_warmup_length()
        self.historical_bars = deque(maxlen=self.num_historical_candles_needed)
        
        # Trading parameters
        self.entry_conf = entry_conf
        self.adx_thresh = adx_thresh
        self.stop_pts = stop_pts 
        self.target_pts = target_pts
        
        self.skip_stats: dict = {'stop_too_tight': 0, 'stop_too_wide': 0, 'predict_error': 0}

        logging.info(f"📊 Strategy: {self.strategy.__class__.__name__}")
        atr_gate = f", MinStopATR={self.min_stop_atr_mult}×ATR" if self.min_stop_atr_mult else ""
        logging.info(f"📈 Trade Params: Entry={self.entry_conf}, ADX={self.adx_thresh}, "
                    f"Stop={self.stop_pts} pts, Target={self.target_pts} pts, "
                    f"HighConf={self.high_conf_multiplier}x @ ≥90%, "
                    f"MinStop={self.min_stop_pts}pts{atr_gate}")

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

    def _update_mfe(self, current_price: float) -> None:
        """Update peak unrealized gain (points) for the open trade."""
        if not self.in_position or self.entry_price is None:
            return
        if self.position_type == 'LONG':
            unrealized = current_price - self.entry_price
        elif self.position_type == 'SHORT':
            unrealized = self.entry_price - current_price
        else:
            return
        if unrealized > self.mfe_pts:
            self.mfe_pts = unrealized

    def _check_and_set_breakeven(self, current_price: float) -> bool:
        """
        Move stop to entry price once the trade reaches 2R profit.

        Activated only when breakeven_on_2r=True AND predicted_rr >= 3.0 at
        entry — ensures there is always a gap between the 2R trigger and TP.
        Fires at most once per trade — subsequent calls are no-ops once
        breakeven_set=True. Returns True on the bar the move is triggered so
        callers can react (e.g. send a broker API call to update the live stop).
        """
        if not self.breakeven_on_2r:
            return False
        predicted_rr = getattr(self.strategy, '_latest_risk_rr', 0.0)
        if predicted_rr < 3.0:
            return False
        if not self.in_position or self.breakeven_set:
            return False
        if self.entry_price is None or self.stop_loss is None:
            return False

        stop_dist = abs(self.entry_price - self.stop_loss)
        if stop_dist == 0:
            return False

        if self.position_type == 'LONG':
            two_r_price = self.entry_price + 2 * stop_dist
            triggered = current_price >= two_r_price
        elif self.position_type == 'SHORT':
            two_r_price = self.entry_price - 2 * stop_dist
            triggered = current_price <= two_r_price
        else:
            return False

        if triggered:
            old_stop = self.stop_loss
            self.stop_loss = self.entry_price
            self.breakeven_set = True
            logging.info(
                f"🔒 Break-even set — stop moved {old_stop:.2f} → {self.entry_price:.2f} "
                f"(2R={two_r_price:.2f} reached, price={current_price:.2f})"
            )
            return True
        return False

    def _reset_position_state(self):
        """Reset position state after exit."""
        self.in_position = False
        self.position_type = None
        self.entry_price = None
        self.stop_loss = None
        self.profit_target = None
        self.stop_orderId = None
        self.limit_orderId = None
        self.stop_bracket_order_id = None
        self.position_size = None
        self.entry_timestamp = None
        self.mfe_pts = 0.0
        self.breakeven_set = False

    def _log_exit(self, exit_price, exit_reason, pnl):
        """Log exit information."""
        target_dist = abs(self.profit_target - self.entry_price) if (self.profit_target and self.entry_price) else 0
        mfe_pct = f" ({self.mfe_pts / target_dist * 100:.0f}% of target)" if target_dist else ""
        print("="*40)
        print(f"🛑 EXIT {self.position_type} @ {exit_price:.2f} ({exit_reason})")
        print(f"  Entry: {self.entry_price:.2f} | PnL Points: {pnl:.2f} | MFE: +{self.mfe_pts:.2f}pts{mfe_pct}")
        print("="*40)
        logging.info(f"EXIT {self.position_type} @ {exit_price:.2f} ({exit_reason}) "
                    f"Entry: {self.entry_price:.2f} | PnL Points: {pnl:.2f} | "
                    f"MFE: +{self.mfe_pts:.2f}pts{mfe_pct}")

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
            
            # Convert timestamp column to datetime, normalize to ET, and set as index
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                if df['timestamp'].dt.tz is None:
                    df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
                df['timestamp'] = df['timestamp'].dt.tz_convert('America/New_York')
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
            bar_time = df.index[-1]

            # Display prediction on every bar regardless of session
            pred_labels = {0: "HOLD", 1: "BUY", 2: "SELL"}
            zone_count = getattr(self.strategy, 'active_zone_count', None)
            zone_str = f" zones={zone_count}" if zone_count is not None else ""
            rr_val = getattr(self.strategy, '_latest_risk_rr', None)
            rr_str = f" rr={rr_val:.2f}" if rr_val is not None and prediction != 0 else ""
            print(f"🤖 AI: {pred_labels[prediction]} (Conf: {confidence:.2%}{rr_str}){zone_str}")
            logging.info(f"AI: {pred_labels[prediction]} (Conf: {confidence:.2%}{rr_str}) ADX: {latest_bar.get('adx', 0):.1f}{zone_str}")

            # Let the strategy veto entries outside its allowed trading window
            if not self.strategy.is_trading_allowed(bar_time):
                return

            # Check if should enter trade
            should_enter, direction = self.strategy.should_enter_trade(
                prediction,
                confidence,
                latest_bar,
                self.entry_conf,
                self.adx_thresh
            )
            
            if should_enter:
                # ✅ CHECK FOR EXISTING POSITION BEFORE ENTERING
                has_position = await self._has_existing_position()
                if has_position:
                    print("⚠️ SIGNAL IGNORED - Already in position")
                    logging.info(f"{direction} signal ignored: Already in position")
                    return
                
                close_price = latest_bar['close']
                tick_size = self._get_tick_size()

                # Let the strategy provide its own stop/target (e.g. zone-based).
                # Fall back to the bot's global stop_pts / target_pts if not provided.
                strat_stop, strat_target = self.strategy.get_stop_target_pts(
                    df, direction, close_price
                )
                stop_pts   = strat_stop   if strat_stop   is not None else self.stop_pts
                target_pts = strat_target if strat_target is not None else self.target_pts

                # On high-confidence signals, extend the target (same risk, larger reward).
                HIGH_CONF_THRESHOLD = 0.90
                if confidence >= HIGH_CONF_THRESHOLD and self.high_conf_multiplier > 1.0:
                    extended_target = target_pts * self.high_conf_multiplier
                    logging.info(
                        f"⚡ High confidence ({confidence:.2%}) — target extended "
                        f"{target_pts:.2f}pts → {extended_target:.2f}pts "
                        f"(×{self.high_conf_multiplier}, risk unchanged)"
                    )
                    target_pts = extended_target

                effective_risk = self.risk_amount

                if stop_pts is None or target_pts is None:
                    logging.error(
                        "❌ No stop/target available — set --stop_pts/--target_pts "
                        "or use a strategy that provides them."
                    )
                    return

                # Dynamic minimum stop: max of fixed floor and ATR-based floor.
                # vty_atr_14 is stored normalized (atr/close); multiply back to get points.
                effective_min_stop_pts = self.min_stop_pts
                if self.min_stop_atr_mult > 0:
                    vty_atr_14 = latest_bar.get('vty_atr_14', 0) or 0
                    atr_pts = vty_atr_14 * close_price
                    atr_floor = atr_pts * self.min_stop_atr_mult
                    effective_min_stop_pts = max(self.min_stop_pts, atr_floor)

                min_stop_ticks = effective_min_stop_pts / tick_size
                stop_ticks_raw = stop_pts / tick_size
                if stop_ticks_raw < min_stop_ticks:
                    gate_desc = (
                        f"ATR-based {effective_min_stop_pts:.2f}pts "
                        f"({self.min_stop_atr_mult}×ATR)"
                        if self.min_stop_atr_mult and effective_min_stop_pts > self.min_stop_pts
                        else f"fixed {self.min_stop_pts}pts"
                    )
                    logging.info(
                        f"⚠️ Signal skipped — zone stop too tight "
                        f"({stop_pts:.2f}pts / {stop_ticks_raw:.1f} ticks < {gate_desc})"
                    )
                    self.skip_stats['stop_too_tight'] += 1
                    return

                if direction == 'LONG':
                    stop_loss        = close_price - stop_pts
                    profit_target    = close_price + target_pts
                    stop_ticks       = -int(stop_pts / tick_size)
                    take_profit_ticks = int(target_pts / tick_size)
                    size = self._calculate_size(stop_ticks, effective_risk)

                    if size == 0:
                        logging.info(
                            f"⚠️ Signal skipped — zone stop {stop_pts:.2f}pts "
                            f"({abs(stop_ticks)} ticks × ${self._get_tick_value():.2f}) "
                            f"exceeds risk_amount ${self.risk_amount:.0f}"
                        )
                        self.skip_stats['stop_too_wide'] += 1
                        return

                    order_result = await self._place_order(
                        side=0,
                        close_price=close_price,
                        stop_loss=stop_loss,
                        profit_target=profit_target,
                        stop_ticks=stop_ticks,
                        take_profit_ticks=take_profit_ticks,
                        size=size
                    )

                    if order_result:
                        print("="*40)
                        print(f"🔥 LONG SIGNAL")
                        print(f"  Reference: {close_price:.2f}")
                        print(f"  Stop: {stop_loss:.2f} | Target: {profit_target:.2f}")
                        print("="*40)
                        logging.info(f"LONG SIGNAL @ {close_price:.2f} Stop: {stop_loss:.2f} Target: {profit_target:.2f}")
                    else:
                        logging.warning(f"⚠️ LONG order rejected by broker @ {close_price:.2f} — signal not placed")

                else:  # SHORT
                    stop_loss        = close_price + stop_pts
                    profit_target    = close_price - target_pts
                    stop_ticks       = int(stop_pts / tick_size)
                    take_profit_ticks = -int(target_pts / tick_size)
                    size = self._calculate_size(stop_ticks, effective_risk)

                    if size == 0:
                        logging.info(
                            f"⚠️ Signal skipped — zone stop {stop_pts:.2f}pts "
                            f"({abs(stop_ticks)} ticks × ${self._get_tick_value():.2f}) "
                            f"exceeds risk_amount ${self.risk_amount:.0f}"
                        )
                        self.skip_stats['stop_too_wide'] += 1
                        return

                    order_result = await self._place_order(
                        side=1,
                        close_price=close_price,
                        stop_loss=stop_loss,
                        profit_target=profit_target,
                        stop_ticks=stop_ticks,
                        take_profit_ticks=take_profit_ticks,
                        size=size
                    )

                    if order_result:
                        print("="*40)
                        print(f"🥶 SHORT SIGNAL")
                        print(f"  Reference: {close_price:.2f}")
                        print(f"  Stop: {stop_loss:.2f} | Target: {profit_target:.2f}")
                        print("="*40)
                        logging.info(f"SHORT SIGNAL @ {close_price:.2f} Stop: {stop_loss:.2f} Target: {profit_target:.2f}")
                    else:
                        logging.warning(f"⚠️ SHORT order rejected by broker @ {close_price:.2f} — signal not placed")
                    
        except Exception as e:
            self.skip_stats['predict_error'] += 1
            logging.exception(f"❌ Error during AI prediction: {e}")
            if "Required inputs" in str(e) and "missing from input feed" in str(e):
                raise RuntimeError(
                    f"Model/strategy input mismatch — {e}. "
                    "Check that --model matches the strategy (e.g. cisd-ote7 requires v7.onnx)."
                ) from e

    @abstractmethod
    def _get_tick_size(self):
        """Get tick size for the contract. Must be implemented by subclass."""
        pass

    @abstractmethod
    def _get_tick_value(self):
        """Get dollar value per tick for the contract. Must be implemented by subclass."""
        pass

    def _calculate_size(self, sl_ticks, effective_risk=None):
        """
        Return dynamic contract size based on risk_amount, or fall back to self.size.
        Pass effective_risk to override risk_amount (e.g. for high-confidence scaling).
        Returns 0 when risk_amount is set but even 1 contract would exceed it
        (caller should skip the signal in that case).
        """
        risk = effective_risk if effective_risk is not None else self.risk_amount
        if risk:
            if sl_ticks == 0:
                return 0  # can't size a 0-tick stop; caller will skip the signal
            tick_value = self._get_tick_value()
            raw = math.floor(risk / (abs(sl_ticks) * tick_value))
            return min(self.max_contracts, raw)  # 0 is preserved — caller handles it
        return self.size

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