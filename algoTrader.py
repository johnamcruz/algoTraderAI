#!/usr/bin/env python3
"""
Real-Time Trading Bot with SignalR - Aggregates ticks into time-based bars.

This version uses an asyncio.Lock and a "watcher" task to ensure bars
are closed by the clock, even if no new ticks arrive. This matches
TradingView's bar-closing behavior.

Requirements:
    pip install onnxruntime pandas pandas-ta signalrcore requests joblib

Usage:
    python algoTrader.py --account YOUR_ACCCOUNT --contract YOUR_CONTRACT --size 1 --username YOUR_USERNAME --apikey YOUR_API_KEY --timeframe 5
"""

#import onnxruntime as ort
import numpy as np
#import pandas as pd
#import pandas_ta as ta
#import joblib
import asyncio
import json
import argparse
import requests
from pysignalr.client import SignalRClient
from datetime import datetime, timedelta, timezone
from collections import deque
import warnings
warnings.filterwarnings('ignore')

MARKET_HUB = "https://rtc.alphaticks.projectx.com/hubs/market"

# =========================================================
# AUTHENTICATION
# =========================================================

def authenticate(username, api_key):
    """
    Authenticate with TopstepX and get JWT token
    
    Returns: JWT token string or None if failed
    """
    auth_url = "https://api.alphaticks.projectx.com/api/Auth/loginKey"
    
    payload = {
        "userName": username,
        "apiKey": api_key
    }
    
    try:
        print("🔐 Authenticating...")
        response = requests.post(auth_url, json=payload, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('success') and data.get('token'):
            print("✅ Authentication successful!")
            return data['token']
        else:
            error_msg = data.get('errorMessage', 'Unknown error')
            print(f"❌ Authentication failed: {error_msg}")
            return None
            
    except Exception as e:
        print(f"❌ Authentication error: {e}")
        return None
    
# =========================================================
# REAL-TIME TRADING BOT CLASS
# =========================================================

class RealTimeBot:
    def __init__(self, token, contract, timeframe_minutes=1):
        self.hub_url = f"{MARKET_HUB}?access_token={token}"
        self.contract = contract
        self.timeframe_minutes = int(timeframe_minutes)
        self.client = SignalRClient(self.hub_url)
                
        self.current_bar = {}
        self.current_bar_time = None
        self.bar_lock = asyncio.Lock()  # Lock to prevent race conditions
        self.closer_task = None         # Handle for the watcher task     
        self.historical_bars = deque(maxlen=100)   
        
        print(f"🤖 Bot initialized for {self.contract} on a {self.timeframe_minutes}-minute timeframe.")

        # Register handlers
        self.client.on_open(self.on_open)
        self.client.on_close(self.on_close)
        self.client.on_error(self.on_error)
        self.client.on("GatewayTrade", self.process_tick)

    #Function to pre-fill the bar history
    async def fetch_historical_data(self):
        """
        Fetches the 100 most recent bars to "prime" the historical_bars deque.
        """
        try:
            print(f"✅ Successfully pre-filled {len(self.historical_bars)} historical bars.")
        except Exception as e:
            print(f"❌ Could not fetch historical data: {e}.")
            print("Bot will start with live data only and build history (will take 100 bars to start trading).")

    async def run(self):
        """
        --- MODIFIED ---
        Starts the bot, the bar closer, and connects to the SignalR hub.
        """
        # Fill history before starting the real-time loop
        await self.fetch_historical_data()

        print("🚀 Starting bot connection...")                
        self.closer_task = asyncio.create_task(self.bar_closer_watcher())        
        await self.client.run()

    async def on_open(self) -> None:
        """Called when the connection is established."""
        print("✅ Connected to market hub")
        try:
            await self.client.send("SubscribeContractTrades", [self.contract])
            print(f"✅ Subscription successful for {self.contract}")
        except Exception as e:
            print(f"❌ Subscription error: {e}")

    async def on_close(self) -> None:
        """
        --- MODIFIED ---
        Called when the connection is closed.
        """
        print('Disconnected from the server')
        if self.closer_task:
            self.closer_task.cancel() # Stop the watcher task

    async def on_error(self, message: str) -> None:
        """Called when a SignalR error occurs."""
        print(f"❌ SignalR Error: {message}")

    async def process_tick(self, data):
        """Processes incoming tick data from the hub."""
        try:
            # If data is like ['CON.F.US.EP.Z25', [ {...} ]]
            if isinstance(data, list) and len(data) >= 2 and isinstance(data[1], list):
                for trade in data[1]:
                    await self.handle_trade(trade)
            # If data is a single dict
            elif isinstance(data, dict):
                await self.handle_trade(data)
            # else:
            #    print("Unexpected data format:", data) # Can be noisy
        except Exception as e:
            print(f"process_tick error: {e} | Data: {data}")
    
    def _get_bar_time(self, ts: datetime) -> datetime:
        """Helper to floor a timestamp to the correct bar interval."""
        # API timestamps may lack timezone, assume UTC
        if ts.tzinfo is None:
             ts = ts.replace(tzinfo=timezone.utc)
        else:
             ts = ts.astimezone(timezone.utc)
        
        # Floor to the minute
        bar_time = ts.replace(second=0, microsecond=0)
        
        # Floor to the timeframe
        if self.timeframe_minutes > 1:
            bar_time = bar_time.replace(
                minute=(bar_time.minute // self.timeframe_minutes) * self.timeframe_minutes
            )
        return bar_time

    async def _close_and_print_bar(self):
        """Internal function to print the bar and reset state. MUST be called inside a lock."""
        if self.current_bar:                        
            print(f"Time:  {self.current_bar_time} O: {self.current_bar['open']} H: {self.current_bar['high']} L: {self.current_bar['low']} C: {self.current_bar['close']} V: {self.current_bar['volume']}")
            
            #TODO Add AI model prediction here
            
            # Reset the bar
            self.current_bar = {}
            self.current_bar_time = None

    async def bar_closer_watcher(self):
        """
        This background task watches the clock. If a bar's time is up,
        it acquires the lock and force-closes the bar, ensuring that
        bars are closed even if no new ticks arrive.
        """
        print("Bar closer watcher started...")
        while True:
            try:
                if not self.current_bar_time:
                    # No active bar, check again in a bit
                    await asyncio.sleep(0.1)
                    continue

                # Calculate when the current bar should end
                next_bar_time = self.current_bar_time + timedelta(minutes=self.timeframe_minutes)
                
                # Get current time (must be offset-aware!)
                now_utc = datetime.now(timezone.utc)
                
                # Calculate how long to sleep
                sleep_duration = (next_bar_time - now_utc).total_seconds()
                
                if sleep_duration > 0:
                    # Sleep until the bar is supposed to end (plus a tiny buffer)
                    await asyncio.sleep(sleep_duration + 0.05) # 50ms buffer

                # --- Time is up, try to close the bar ---
                async with self.bar_lock:
                    # Check if the bar is *still* the one we were waiting for.
                    # It might have *already* been closed by a fast-moving tick.
                    if self.current_bar and self.current_bar_time < next_bar_time:                        
                        await self._close_and_print_bar()

            except asyncio.CancelledError:
                print("Bar closer watcher stopping.")
                break
            except Exception as e:
                print(f"Error in bar_closer_watcher: {e}")
                await asyncio.sleep(1) # Don't spam errors


    async def handle_trade(self, trade):
        """
        --- MODIFIED ---
        Aggregates a single trade tick into a time-based bar.
        Now uses a lock to coordinate with the bar_closer_watcher.
        """
        try:
            ts = datetime.fromisoformat(trade.get("timestamp"))
            price = trade.get("price")
            volume = trade.get("volume", 0)

            if price is None:
                return

            # Calculate which bar this tick belongs to
            bar_time = self._get_bar_time(ts)

            # --- Acquire lock to modify bar state ---
            async with self.bar_lock:
                # Check if this tick starts a new bar
                if bar_time != self.current_bar_time:
                    # --- 1. A new bar is starting ---
                    
                    # First, close and print the *previous* bar, if it exists
                    # (The timer task might have already done this, but this is a safety catch)
                    if self.current_bar:
                        print(f"[Tick] Closing bar for {self.current_bar_time}")
                        await self._close_and_print_bar()
                    
                    # Initialize the new bar
                    self.current_bar_time = bar_time
                    self.current_bar = {
                        "timestamp": bar_time,
                        "open": price,
                        "high": price,
                        "low": price,
                        "close": price,
                        "volume": volume
                    }
                else:
                    # --- 2. This tick updates the current bar ---
                    if not self.current_bar:
                        # This can happen if the timer *just* closed a bar
                        # We'll just re-initialize.
                        self.current_bar_time = bar_time
                        self.current_bar = {"timestamp": bar_time, "open": price, "high": price, "low": price, "close": price, "volume": volume}
                    else:
                        self.current_bar["high"] = max(self.current_bar["high"], price)
                        self.current_bar["low"] = min(self.current_bar["low"], price)
                        self.current_bar["close"] = price
                        self.current_bar["volume"] += volume
            
        except Exception as e:
            print(f"handle_trade error: {e} | Trade: {trade}")


# =========================================================
# MAIN
# =========================================================

def main():
    parser = argparse.ArgumentParser(description='Real-Time Futures Trading Bot')
    parser.add_argument('--account', type=str, required=True,
                        help='TopstepX account to trade')
    parser.add_argument('--contract', type=str, required=True,
                        help='Contract to trade ie. CON.F.US.ENQ.U25')
    parser.add_argument('--size', type=int, required=True,
                        help='Contract size')
    parser.add_argument('--username', type=str, required=True,
                        help='TopstepX username')
    parser.add_argument('--apikey', type=str, required=True,
                        help='TopstepX API key')
    parser.add_argument('--model', type=str, default='lstm_model.onnx',
                        help='Path to ONNX model')
    # --- ADDED ARGUMENT ---
    parser.add_argument('--timeframe', type=int, choices=[1, 3, 5], default=1,
                        help='Timeframe in minutes to aggregate ticks (1, 3, 5)')
    
    args = parser.parse_args()
    
    # Authenticate and get JWT token
    jwt_token = authenticate(args.username, args.apikey)

    if not jwt_token:
        print("\n❌ Cannot start bot without valid authentication")
        return
    
    print(f"\n🎫 Token received (expires in ~24 hours)")
    
    try:        
        bot = RealTimeBot(jwt_token, args.contract, args.timeframe)        
        asyncio.run(bot.run())        
    except KeyboardInterrupt:
        print("\n👋 Bot stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()