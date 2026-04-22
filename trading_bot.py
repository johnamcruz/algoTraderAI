#!/usr/bin/env python3
"""
Real time Bot Engine - which manages all live trading 
operations. Its primary responsibilities are:

1.  Connecting to the exchange API (SignalR for ticks, REST for orders/history).
2.  Aggregating real-time tick data into time-based (e.g., 3-min) OHLCV bars.
3.  Calling a pluggable AI strategy (from BaseStrategy) on each bar close.
4.  Executing trades and managing position state with ATR-based stop-loss 
    and profit-target brackets.
"""

import asyncio
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from pysignalr.client import SignalRClient
from strategy_base import BaseStrategy
from bot_utils import parse_future_symbol
from trading_bot_base import TradingBot

# =========================================================
# REAL-TIME TRADING BOT CLASS
# =========================================================
class RealTimeBot(TradingBot):
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
        stop_pts,
        target_pts,
        enable_trailing_stop=False,
        risk_amount=None,
        high_conf_multiplier=1.0,
        max_contracts=15,
        min_stop_pts=1.0,
        min_stop_atr_mult=0.5,
        breakeven_on_1r=True,
    ):
        """
        Initialize the real-time trading bot.
        
        Args:
            token: Authentication token
            market_hub: Market hub URL
            base_url: Base API URL
            account: Trading account ID
            contract: Contract ID
            size: Position size
            timeframe_minutes: Bar timeframe
            strategy: Strategy instance (implements BaseStrategy)
            entry_conf: Minimum confidence for entry
            adx_thresh: Minimum ADX for entry
            stop_pts: Stop loss in fixed points
            target_pts: Profit target in fixed points
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
            enable_trailing_stop=enable_trailing_stop,
            risk_amount=risk_amount,
            high_conf_multiplier=high_conf_multiplier,
            max_contracts=max_contracts,
            min_stop_pts=min_stop_pts,
            min_stop_atr_mult=min_stop_atr_mult,
            breakeven_on_1r=breakeven_on_1r,
        )

        # Real-time specific attributes
        self.hub_url = f"{market_hub}?access_token={token}"
        self.base_url = base_url
        self.account = account
        self.client = SignalRClient(self.hub_url)
        self.token = token
        
        self.current_bar = {}
        self.current_bar_time = None
        self.bar_lock = asyncio.Lock()
        self.closer_task = None
        self.contracts = None
        
        print(f"🤖 Bot initialized for {self.contract} on {self.timeframe_minutes}-min timeframe.")
        print(f"📈 Trade Params: Entry={self.entry_conf}, ADX={self.adx_thresh}, "
              f"Stop={self.stop_pts} pts Target={self.target_pts} pts")
        print(f"📊 Strategy: {self.strategy.__class__.__name__}")
        
        # Register handlers
        self.client.on_open(self.on_open)
        self.client.on_close(self.on_close)
        self.client.on_error(self.on_error)
        self.client.on("GatewayTrade", self.process_tick)

    # =========================================================
    # BAR TIME CALCULATION (RealTimeBot specific)
    # =========================================================
    def _get_bar_time(self, ts):
        """
        Rounds down a timestamp to the nearest timeframe boundary.
        Used for aggregating ticks into bars.
        
        Args:
            ts: datetime object
            
        Returns:
            datetime: Rounded bar time
        """
        minute = (ts.minute // self.timeframe_minutes) * self.timeframe_minutes
        return ts.replace(minute=minute, second=0, microsecond=0)

    # =========================================================
    # CONNECTION HANDLERS
    # =========================================================
    async def on_open(self):
        """Callback when connection opens."""
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
    # CONTRACTS FETCHING
    # =========================================================
    async def fetch_contract_data(self):
        """Fetches contracts."""
        contracts_url = f"{self.base_url}/Contract/available"
        payload = { "live": False }
        headers = {'Authorization': f'Bearer {self.token}'}
        try:
            response = requests.post(contracts_url, headers=headers, json=payload, timeout=10)            
            response.raise_for_status()
            self.contracts = response.json().get('contracts', [])
            logging.debug(self.contracts)
            logging.info("✅ Successfully retrieve contracts")
        except Exception as e:
            logging.exception(f"❌ Could not fetch historical data: {e}.")

    def find_contract(self, contract_id):
        """Find contract in contracts array"""
        for item in self.contracts:
            if item.get('id') == contract_id:
                return item
        return None             

    # =========================================================
    # HISTORICAL DATA FETCHING
    # =========================================================
    async def fetch_historical_data(self):
        """Fetches recent bars to prime the historical data deque"""
        historical_url = f"{self.base_url}/History/retrieveBars"
        end_time_dt = datetime.now(timezone.utc).replace(microsecond=0)
        # Fetch a bit more history (e.g., 3 days) to ensure enough warmup for indicators        
        start_time_dt = end_time_dt - timedelta(days=3)
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
        logging.debug(payload)
        headers = {'Authorization': f'Bearer {self.token}'}
        try:
            response = requests.post(historical_url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            bars_fetched = 0

            bars_from_api = response.json().get('bars', [])
            bars_from_api.reverse()
            logging.debug(bars_from_api)
            
            for bar in bars_from_api:
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
    # BAR PROCESSING
    # =========================================================
    def _get_tick_size(self):
        """Get tick size from contract details."""
        contract_details = self.find_contract(self.contract)
        if contract_details and contract_details.get('tickSize'):
            return contract_details['tickSize']
        return 0.01

    def _get_tick_value(self):
        """Get dollar value per tick from contract details."""
        contract_details = self.find_contract(self.contract)
        if contract_details and contract_details.get('tickValue'):
            return contract_details['tickValue']
        return 0.50

    async def _has_existing_position(self):
        """
        Check if there's an existing position by querying the broker API.
        
        Returns:
            bool: True if position exists, False otherwise
        """
        try:
            # Call broker API to get open positions
            positions_url = f"{self.base_url}/Position/searchOpen"  # ✅ Correct endpoint
            headers = {'Authorization': f'Bearer {self.token}'}
            payload = {"accountId": self.account}
            
            response = requests.post(
                positions_url, 
                headers=headers, 
                json=payload, 
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Check if API call was successful
            if not data.get('success', False):
                error_msg = data.get('errorMessage', 'Unknown error')
                logging.error(f"❌ Position check API error: {error_msg}")
                return True  # Fail-safe
            
            positions = data.get('positions', [])
            
            # Check if any position matches our contract
            for position in positions:
                if position.get('contractId') == self.contract:
                    size = position.get('size', 0)  # ✅ Using 'size' field
                    if size != 0:
                        logging.info(f"📍 Existing position found: {size} contracts @ {position.get('averagePrice', 0):.2f}")
                        return True
            
            return False
            
        except Exception as e:
            logging.error(f"❌ Error checking positions: {e}")
            # Fail-safe: Assume position exists to prevent duplicate orders
            return True

    async def _close_and_print_bar(self):
        """Finalize current bar and run strategy."""
        if not self.current_bar:
            return
        
        self.historical_bars.append(self.current_bar)
        
        bar_time_str = datetime.fromisoformat(
            self.current_bar["timestamp"]
        ).astimezone(None).strftime("%Y-%m-%d %H:%M")
        
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
            await self._run_ai_prediction()  # Use base class method
        else:
            logging.info(f"⏳ Waiting for more bars... "
                  f"({len(self.historical_bars)}/{self.num_historical_candles_needed})")
        
        # Reset bar state
        self.current_bar = {}


    # =========================================================
    # TICK PROCESSING
    # =========================================================
    async def process_tick(self, data):
        """Process incoming tick data."""        
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
    # ORDER MANAGEMENT
    # =========================================================
    async def _place_order(self, side, close_price, stop_loss, profit_target, stop_ticks, take_profit_ticks, size):
        """
        Place order to broker immediately.

        Args:
            side: 0 for LONG, 1 for SHORT
            close_price: Reference price (not used in live bot)
            stop_loss: Stop loss price (not used - calculated by broker from ticks)
            profit_target: Profit target price (not used - calculated by broker from ticks)
            stop_ticks: Stop loss in ticks
            take_profit_ticks: Profit target in ticks
            size: Number of contracts (dynamically calculated or static fallback)
        """
        order_url = f"{self.base_url}/Order/place"
        payload = {
            "accountId": self.account,
            "contractId": self.contract,
            "type": 2,  # Market order
            "side": side,
            "size": size,
            "stopLossBracket": {
                "ticks": stop_ticks,
                "type": 5 if self.enable_trailing_stop else 4
            },
            "takeProfitBracket": {
                "ticks": take_profit_ticks,
                "type": 1
            }
        }        
        logging.debug(payload)
        headers = {'Authorization': f'Bearer {self.token}'}
        try:
            response = requests.post(order_url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()
            logging.debug(data)
            if data.get('success') and data.get('orderId'):
                order_id = data.get('orderId')
                logging.info(f"✅ Order placed successfully: {order_id}")

                # Set position state so tick-level breakeven checks fire
                self.in_position = True
                self.position_type = 'LONG' if side == 0 else 'SHORT'
                self.entry_price = close_price
                self.stop_loss = stop_loss
                self.profit_target = profit_target
                self.position_size = size

                # Give broker time to create bracket orders, then store stop ID
                await asyncio.sleep(2.0)
                self.stop_bracket_order_id = self._fetch_stop_bracket_order_id()

                return order_id
            else:
                logging.exception(f"❌ Order failed: {data.get('errorMessage')}")
                return None
        except Exception as e:
            logging.exception(f"❌ Could not place order: {e}.")
            return None

    def _fetch_stop_bracket_order_id(self):
        """Search open orders and return the stop bracket order ID for our contract."""
        try:
            r = requests.post(
                f"{self.base_url}/Order/searchOpen",
                headers={'Authorization': f'Bearer {self.token}'},
                json={"accountId": self.account},
                timeout=10
            )
            r.raise_for_status()
            data = r.json()
            if not data.get('success'):
                logging.error(f"❌ searchOpen failed: {data.get('errorMessage')}")
                return None
            for order in data.get('orders', []):
                if order.get('contractId') == self.contract and order.get('type') == 4:
                    oid = order['id']
                    logging.info(f"📌 Stop bracket order ID: {oid} @ stopPrice={order.get('stopPrice')}")
                    return oid
            logging.warning("⚠️ No stop bracket order found after entry")
            return None
        except Exception as e:
            logging.error(f"❌ Error in searchOpen: {e}")
            return None

    def _modify_order(self, order_id: int, stop_price: float, size: int) -> bool:
        """Move an existing order's stop price via Order/modify.

        Works on OCO bracket stops (confirmed via live API test) even though
        not documented. OCO linkage is preserved. Returns True on success.

        Kept separate from _on_breakeven_triggered so future features
        (e.g. trailing stop adjustments) can reuse the same API call.
        """
        headers = {'Authorization': f'Bearer {self.token}'}
        try:
            r = requests.post(
                f"{self.base_url}/Order/modify",
                headers=headers,
                json={
                    "accountId":  self.account,
                    "orderId":    order_id,
                    "size":       size,
                    "stopPrice":  stop_price,
                    "limitPrice": None,
                    "trailPrice": None,
                },
                timeout=10
            )
            r.raise_for_status()
            result = r.json()
            if result.get('success'):
                logging.info(f"✅ Order {order_id} stop moved to {stop_price:.2f}")
                return True
            logging.error(
                f"❌ Order/modify failed for order {order_id}: {result.get('errorMessage')}"
            )
            return False
        except Exception as e:
            logging.error(f"❌ Error calling Order/modify for order {order_id}: {e}")
            return False

    async def _on_breakeven_triggered(self):
        """Move the bracket stop to entry price once 1R profit is reached."""
        if not self.stop_bracket_order_id:
            logging.warning(
                f"⚠️ Breakeven triggered but stop bracket ID unknown — "
                f"broker stop NOT moved to {self.entry_price:.2f}"
            )
            return

        self._modify_order(
            order_id=self.stop_bracket_order_id,
            stop_price=self.entry_price,
            size=self.position_size,
        )

    # =========================================================
    # BAR CLOSER WATCHER
    # =========================================================
    async def bar_closer_watcher(self):
        """Background task to watch the clock and force-close bars."""
        logging.info("⏳ Bar closer watcher started...")
        
        # Wait for the first bar to be created by a tick
        while not self.current_bar_time:
            await asyncio.sleep(0.1)
            
        while True:
            try:
                # Store the time of the bar we are currently watching
                bar_we_are_watching = self.current_bar_time
                
                # Calculate when this bar should close (which is when the *next* bar starts)
                next_bar_start_time = bar_we_are_watching + timedelta(
                    minutes=self.timeframe_minutes
                )
                
                now_utc = datetime.now(timezone.utc)
                sleep_duration = (
                    next_bar_start_time - now_utc
                ).total_seconds()
                
                if sleep_duration > 0:
                    # Sleep until 50ms *after* the bar should have closed
                    await asyncio.sleep(sleep_duration + 0.05) 
                
                # Now that we've slept, check if the bar needs closing
                async with self.bar_lock:
                    # CRITICAL CHECK:
                    # Is the bar we were watching (bar_we_are_watching)
                    # STILL the self.current_bar_time?
                    if self.current_bar and self.current_bar_time == bar_we_are_watching:
                        # YES. This means no ticks came in to close it.
                        # The watcher must close it.
                        logging.info(f"⏰ Watcher closing bar for {bar_we_are_watching}")
                        
                        # Close the bar (which prints and resets self.current_bar={})
                        await self._close_and_print_bar()
                        
                        # CRITICAL FIX: Manually advance the clock
                        # Set the time to the *next* bar interval, 
                        # so the loop can continue watching for the *next* close.
                        self.current_bar_time = next_bar_start_time 
                    
                    # ELSE:
                    # handle_trade() already received a tick for the next bar
                    # and self.current_bar_time has been updated.
                    # The watcher does nothing and loops to watch the new bar.
                        
            except asyncio.CancelledError:
                logging.info("Bar closer watcher stopping.")
                break
            except Exception as e:
                logging.exception(f"Error in bar_closer_watcher: {e}")
                await asyncio.sleep(1) # Wait 1s on error

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
            
            # Check exits if in position (for sim bot compatibility)
            if self.in_position:
                self._update_mfe(price)
                if self._check_and_set_breakeven(price):
                    await self._on_breakeven_triggered()
                exit_price, exit_reason = self._check_exit_conditions(price)

                if exit_reason:
                    pnl = self._calculate_pnl(exit_price)
                    self._log_exit(exit_price, exit_reason, pnl)
                    self._reset_position_state()
            
            # Bar aggregation
            bar_time = self._get_bar_time(ts)
            async with self.bar_lock:
                if bar_time != self.current_bar_time:
                    # Discard stale ticks for bars already closed by the watcher
                    if self.current_bar_time and bar_time < self.current_bar_time:
                        return
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
    # MAIN RUN LOOP
    # =========================================================
    async def run(self):
        """Starts the bot."""
        await self.fetch_historical_data()
        await self.fetch_contract_data()

        contract_details = self.find_contract(self.contract)
        contract_symbol = None        
        if contract_details and contract_details.get('name'):            
            full_contract_name = contract_details['name']
            contract_symbol = parse_future_symbol(full_contract_name)            
            logging.info(f"Identified Contract Symbol: {contract_symbol} from name: {full_contract_name}")            
        else:                        
            logging.error("⚠️ Could not find full contract name via API. ")
            return
        
        # Initialize the strategy with the derived symbol
        self.strategy.set_contract_symbol(contract_symbol)

        self.strategy.load_model()

        print("🚀 Starting bot connection...")
        # ⭐️ CRITICAL FIX: Create the watcher task so it runs on the loop
        self.closer_task = asyncio.create_task(self.bar_closer_watcher())
        
        # Now, await the main blocking client loop
        await self.client.run()
        
        # If the client.run() ever exits (e.g., disconnect), cancel the watcher
        if self.closer_task:
            self.closer_task.cancel()