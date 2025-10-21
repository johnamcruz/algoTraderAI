#!/usr/bin/env python3
"""
Real-Time Trading Bot with SignalR - Aggregates ticks into time-based bars.

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
from datetime import datetime, timedelta
from collections import defaultdict
import threading
import time
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
        
        # State for bar aggregation
        self.current_bar = {}
        self.current_bar_time = None
        
        print(f"🤖 Bot initialized for {self.contract} on a {self.timeframe_minutes}-minute timeframe.")

        # Register handlers
        self.client.on_open(self.on_open)
        self.client.on_close(self.on_close)
        self.client.on_error(self.on_error)
        self.client.on("GatewayTrade", self.process_tick)

    async def run(self):
        """Starts the bot and connects to the SignalR hub."""
        print("🚀 Starting bot connection...")
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
        """Called when the connection is closed."""
        print('Disconnected from the server')

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
            else:
                print("Unexpected data format:", data)
        except Exception as e:
            print(f"process_tick error: {e} | Data: {data}")

    async def handle_trade(self, trade):
        """Aggregates a single trade tick into a time-based bar."""
        try:
            ts = datetime.fromisoformat(trade.get("timestamp"))
            price = trade.get("price")
            volume = trade.get("volume", 0)

            if price is None:
                return # Skip trades with no price

            # --- Core Resampling Logic ---
            # Floor the timestamp to the start of the bar interval
            bar_time = ts.replace(second=0, microsecond=0)
            if self.timeframe_minutes > 1:
                # Calculate the minute flooring
                # e.g., 10:04 with timeframe=3 -> (4 // 3) * 3 = 1 * 3 = 3 -> 10:03
                # e.g., 10:05 with timeframe=3 -> (5 // 3) * 3 = 1 * 3 = 3 -> 10:03
                # e.g., 10:06 with timeframe=3 -> (6 // 3) * 3 = 2 * 3 = 6 -> 10:06
                bar_time = bar_time.replace(minute=(bar_time.minute // self.timeframe_minutes) * self.timeframe_minutes)

            # Check if this tick starts a new bar
            if bar_time != self.current_bar_time:
                # --- 1. A new bar is starting ---
                
                # If a previous bar exists, print it as "closed"
                if self.current_bar:                    
                    print(f"Time: {self.current_bar_time} O: {self.current_bar['open']} H: {self.current_bar['high']} L: {self.current_bar['low']} C: {self.current_bar['close']} V: {self.current_bar['volume']}")                    
                
                # Initialize the new bar
                self.current_bar_time = bar_time
                self.current_bar = {
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": volume
                }
            else:
                # --- 2. This tick updates the current bar ---
                self.current_bar["high"] = max(self.current_bar["high"], price)
                self.current_bar["low"] = min(self.current_bar["low"], price)
                self.current_bar["close"] = price
                self.current_bar["volume"] += volume
            
            # Optional: Print live-updating bar (can be noisy)
            # print(f"Live Bar ({self.current_bar_time}): C: {self.current_bar['close']} V: {self.current_bar['volume']}", end="\r")

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