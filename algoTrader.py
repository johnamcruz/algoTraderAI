#!/usr/bin/env python3
"""
Real-Time Trading Bot with SignalR - Listens to live bar updates

Requirements:
    pip install onnxruntime pandas pandas-ta signalrcore requests joblib

Usage:
    python algoTrader.py --account YOUR_ACCCOUNT --contract --username YOUR_USERNAME --apikey YOUR_API_KEY
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
        print("🔐 Authenticating with TopstepX...")
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
# REAL-TIME TRADING BOT
# =========================================================

bars = defaultdict(lambda: {"open": None, "high": 0, "low": float("inf"), "close": None, "volume": 0})

async def process_tick(data):
    try:
        # If data is like ['CON.F.US.EP.Z25', [ {...} ]]
        if isinstance(data, list) and len(data) >= 2 and isinstance(data[1], list):
            for trade in data[1]:
                await handle_trade(trade)
        # If data is a single dict
        elif isinstance(data, dict):
            await handle_trade(data)
        else:
            print("Unexpected data format:", data)
    except Exception as e:
        print("process_tick error:", e)

async def handle_trade(trade):    
    ts = datetime.fromisoformat(trade.get("timestamp"))
    price = trade.get("price")
    volume = trade.get("volume", 0)

    bar = bars[ts]
    if bar["open"] is None:
        bar["open"] = price
    bar["high"] = max(bar["high"], price)
    bar["low"] = min(bar["low"], price)
    bar["close"] = price
    bar["volume"] += volume

    print(f"{ts}: {bar}")

async def setupSignalR(token, contract):
    hub_url = f"{MARKET_HUB}?access_token={token}"
        
    client = SignalRClient(hub_url)
    
    async def on_open() -> None:
        print("✅ Connected to market hub")        
        try:
            await client.send("SubscribeContractTrades", [contract])     
            print("Subscription successful")       
        except Exception as e:
            print(f"❌ Subscription error: {e}")

    
    async def on_close() -> None:
        print('Disconnected from the server')
        await client.send("UnsubscribeContractTrades", [contract])

    async def on_error(message: str) -> None:
        print(f"❌ SignalR Error: {message}")

    client.on_open(on_open)
    client.on_close(on_close)
    client.on_error(on_error)  # Add error handler
    client.on("GatewayTrade", process_tick)    

    # Just run the client - subscription happens in on_open
    await client.run()


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
    
    args = parser.parse_args()
    
    # Authenticate and get JWT token
    jwt_token = authenticate(args.username, args.apikey)

    if not jwt_token:
        print("\n❌ Cannot start bot without valid authentication")
        return
    
    print(f"\n🎫 Token received (expires in ~24 hours)")
    
    try:
        asyncio.run(setupSignalR(jwt_token, args.contract))
    except KeyboardInterrupt:
        print("\n👋 Bot stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()