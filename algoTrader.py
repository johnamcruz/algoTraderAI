#!/usr/bin/env python3
"""
Real-Time Trading Bot with SignalR - Aggregates ticks and runs AI model.

This version integrates an ONNX model to generate real-time trade signals
based on the winning parameters from backtesting.

Requirements:
    pip install onnxruntime pandas pandas-ta signalrcore requests pickle numpy

Usage:
    python algoTrader.py --account YOUR_ACCCOUNT --contract CON.F.US.ES.Z25 --size 1 --username YOUR_USERNAME --apikey YOUR_API_KEY --timeframe 5 --model "C:\path\to\SUPERTRADER_strategy_1_v2.onnx" --scaler "C:\path\to\SUPERTRADER_scalers_v2.pkl"
"""

import numpy as np
import asyncio
import json
import argparse
import requests
from pysignalr.client import SignalRClient
from datetime import datetime, timedelta, timezone
from dateutil.relativedelta import relativedelta
from collections import deque
import warnings
import os
import pickle

# --- AI & Data Imports ---
import onnxruntime
import pandas as pd
import pandas_ta as ta

warnings.filterwarnings('ignore')

MARKET_HUB = "https://rtc.alphaticks.projectx.com/hubs/market"
BASE_URL = "https://api.alphaticks.projectx.com/api"

# =========================================================
# AUTHENTICATION
# =========================================================

def authenticate(username, api_key):
    """
    Authenticate with TopstepX and get JWT token
    
    Returns: JWT token string or None if failed
    """
    auth_url = f"{BASE_URL}/Auth/loginKey"
    
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
    def __init__(self, token, contract, timeframe_minutes, model_path, scaler_path):
        self.hub_url = f"{MARKET_HUB}?access_token={token}"
        self.contract = contract
        self.timeframe_minutes = int(timeframe_minutes)
        self.client = SignalRClient(self.hub_url)
        self.token = token
                
        self.current_bar = {}
        self.current_bar_time = None
        self.bar_lock = asyncio.Lock()  # Lock to prevent race conditions
        self.closer_task = None         # Handle for the watcher task     
        
        # --- AI & Data Properties ---
        self.num_historical_candles_needed = 60  # seq_len for the AI model
        self.historical_bars = deque(maxlen=60) # Store 100 to allow for indicator warmup
        self.model_path = model_path
        self.scaler_path = scaler_path
        
        self.ort_session = None
        self.scalers = None
        self.es_scaler = None # We are testing ES
        
        # The 9 "lite" features the model was trained on
        self.feature_cols = [
            'compression_level', 'squeeze_duration', 'bb_expanding',
            'atr_expanding', 'price_in_range', 'rsi',
            'compressed_momentum', 'vol_surge', 'body_strength',
        ]
        
        # The winning parameters from the backtest
        self.TRADE_PARAMS = {
            'CONTRACT': 'ES',
            'ENTRY_CONF': 0.55,
            'ADX_THRESH': 20,
            'STOP_ATR_MULT': 2.5
        }
        
        print(f"🤖 Bot initialized for {self.contract} on a {self.timeframe_minutes}-minute timeframe.")
        
        # Load AI model and scalers
        self.load_model_and_scalers()

        # Register handlers
        self.client.on_open(self.on_open)
        self.client.on_close(self.on_close)
        self.client.on_error(self.on_error)
        self.client.on("GatewayTrade", self.process_tick)

    def load_model_and_scalers(self):
        """
        Loads the ONNX model and the scaler .pkl file from disk.
        """
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            if not os.path.exists(self.scaler_path):
                raise FileNotFoundError(f"Scaler file not found: {self.scaler_path}")
            
            # 1. Load ONNX Model
            self.ort_session = onnxruntime.InferenceSession(self.model_path)
            print(f"✅ Successfully loaded ONNX model from: {self.model_path}")

            # 2. Load Scalers
            with open(self.scaler_path, 'rb') as f:
                self.scalers = pickle.load(f)
            
            # 3. Extract the ES scaler (since we are testing ES)
            # Make sure your contract key 'ES' matches what's in the .pkl file
            scaler_key = self.TRADE_PARAMS['CONTRACT']
            if self.scalers and scaler_key in self.scalers:
                self.es_scaler = self.scalers[scaler_key]
                print(f"✅ Successfully loaded and extracted '{scaler_key}' scaler.")
            else:
                raise ValueError(f"'{scaler_key}' scaler not found in {self.scaler_path}")

        except Exception as e:
            print(f"❌ CRITICAL ERROR loading model/scalers: {e}")
            print("Bot cannot make predictions. Exiting.")
            # In a real app, you might want to stop the bot here
            raise e # Stop the bot


    def add_ai_features(self, df):
        """
        Calculates all necessary AI features and trade filters on the historical dataframe.
        """
        # Ensure correct dtypes
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)

        # === 1. Calculate base indicators ===
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        
        bbands = ta.bbands(df['close'], length=20, std=2)
        df['bb_upper'] = bbands['BBU_20_2.0']
        df['bb_lower'] = bbands['BBL_20_2.0']
        df['bb_mid'] = bbands['BBM_20_2.0']
        
        keltner = ta.kc(df['high'], df['low'], df['close'], length=20, mamode='ema', atr_length=10, multiplier=2)
        df['kc_upper'] = keltner['KCUe_20_10_2.0']
        df['kc_lower'] = keltner['KCLe_20_10_2.0']
        
        df['rsi'] = ta.rsi(df['close'], length=14)
        df['roc'] = ta.roc(df['close'], length=10)
        
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
        df['adx'] = adx_df['ADX_14'] # Used for trade filter
        
        df['vol_ma'] = df['volume'].rolling(20).mean()

        # === 2. Calculate AI features (the 9 "lite" features) ===
        df['squeeze_on'] = ((df['bb_lower'] > df['kc_lower']) & (df['bb_upper'] < df['kc_upper'])).astype(float)
        
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
        df['bb_width_pct'] = df['bb_width'] / df['bb_width'].rolling(50).mean()
        df['compression_level'] = 1 - df['bb_width_pct'].clip(0, 2)
        
        df['squeeze_duration'] = df['squeeze_on'].groupby(
            (df['squeeze_on'] != df['squeeze_on'].shift()).cumsum()
        ).cumsum()
        
        df['bb_expanding'] = (df['bb_width'].diff(3) > 0).astype(float)
        df['atr_expanding'] = (df['atr'].pct_change(3) > 0.03).astype(float)
        
        df['price_in_range'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']).replace(0, 1)
        
        df['compressed_momentum'] = df['roc'] * df['squeeze_on']
        
        df['vol_ratio'] = df['volume'] / df['vol_ma'].replace(0, 1)
        df['vol_surge'] = (df['vol_ratio'] > 1.3).astype(float)
        
        df['body'] = (df['close'] - df['open']) / df['atr'].replace(0, 1)
        df['body_strength'] = abs(df['body'])
        
        # Clean up NaNs
        df.fillna(method='ffill', inplace=True)
        df.fillna(0, inplace=True) # Fill any remaining NaNs at the beginning
        
        return df


    def get_ai_prediction(self):
        """
        Converts historical bars to a DataFrame, calculates features,
        and runs the ONNX model to get a prediction.
        
        Returns: (down_prob, up_prob, last_bar_data) or (None, None, None)
        """
        # 1. Check if we have enough data (60 bars) to feed the model
        if len(self.historical_bars) < self.num_historical_candles_needed:
            print(f"⏳ Warming up... {len(self.historical_bars)}/{self.num_historical_candles_needed} bars.")
            return None, None, None

        # 2. Convert deque of dicts to DataFrame
        # We use the full deque (100 bars) to ensure indicators are warm
        df = pd.DataFrame(list(self.historical_bars))
        
        # 3. Calculate all features
        df_with_features = self.add_ai_features(df)
        
        # 4. Get the *last 60 bars* of features for the model
        last_60_bars_features = df_with_features.tail(self.num_historical_candles_needed)
        
        # 5. Get the *very last bar's* data for our trade filters (ADX, Squeeze, etc.)
        last_bar_data = df_with_features.iloc[-1]
        
        # 6. Check for NaNs (should be handled, but as a safeguard)
        if last_60_bars_features[self.feature_cols].isnull().values.any():
            print("⏳ Indicators still warming up (NaNs found). Skipping prediction.")
            return None, None, None

        # 7. Get feature values in the correct order
        features_array = last_60_bars_features[self.feature_cols].values
        
        # 8. Scale the data (CRITICAL)
        scaled_features = self.es_scaler.transform(features_array)
        
        # 9. Format for the model (batch_size, seq_len, num_features)
        model_input = np.array([scaled_features], dtype=np.float32)

        # 10. Run prediction
        ort_inputs = {self.ort_session.get_inputs()[0].name: model_input}
        ort_outs = self.ort_session.run(None, ort_inputs)
        
        # ort_outs[0] contains the raw logits, e.g., [[-0.23, 0.45]]
        logits = ort_outs[0][0] 
        
        # 11. Convert to probabilities (softmax)
        probs = np.exp(logits) / np.sum(np.exp(logits))
        
        down_conf = probs[0]
        up_conf = probs[1]
        
        return down_conf, up_conf, last_bar_data


    #Function to pre-fill the bar history
    async def fetch_historical_data(self):
        """
        Fetches the most recent bars to "prime" the historical_bars deque.
        """

        historical_url = f"{BASE_URL}/History/retrieveBars"        
        end_time_dt = datetime.now(timezone.utc).replace(microsecond=0)
        start_time_dt = end_time_dt - relativedelta(days=2) # Get 2 days to be safe
        end_time_str = end_time_dt.isoformat().replace('+00:00', 'Z')
        start_time_str = start_time_dt.isoformat().replace('+00:00', 'Z')
            
        payload = {            
            "contractId": self.contract,
            "live": False,
            "startTime": start_time_str,
            "endTime": end_time_str,
            "unit": 2, # minutes 
            "unitNumber": self.timeframe_minutes,
            # Fetch 100 bars for indicator warmup
            "limit": self.historical_bars.maxlen, 
            "includePartialBar": False
        }
        headers = {
            'Authorization': f'Bearer {self.token}'
        }        
        try:            
            response = requests.post(historical_url,headers=headers, json=payload, timeout=10)
            response.raise_for_status()
        
            data = response.json()
            for bar in data.get('bars', []):                
                formatted_bar = {
                    # Convert timestamp from string to datetime object, then to isoformat
                    "timestamp": datetime.fromisoformat(bar['t']).isoformat(),
                    "open": bar['o'],
                    "high": bar['h'],
                    "low": bar['l'],
                    "close": bar['c'],
                    "volume": bar['v']
                }                
                self.historical_bars.append(formatted_bar)
            
            print(f"✅ Successfully pre-filled {len(self.historical_bars)} historical bars.")
        except Exception as e:
            print(f"❌ Could not fetch historical data: {e}.")
            print(f"Bot will start with live data only and build history (will take {self.historical_bars.maxlen} bars to start trading).")

    async def run(self):
        """        
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
        """
        Internal function to print the bar, run AI, and reset state.
        MUST be called inside a lock.
        """
        if self.current_bar:                        
            print(f"Bar Close: {self.current_bar}")            

            # Make a copy for processing, as self.current_bar will be reset
            closed_bar_data = self.current_bar.copy()            
            self.historical_bars.append(closed_bar_data)
            
            # --- RUN AI PREDICTION ---
            if self.ort_session and self.es_scaler:
                try:
                    down_prob, up_prob, last_bar_data = self.get_ai_prediction()
                    
                    if down_prob is not None:
                        print(f"🧠 AI PREDICTION [ {last_bar_data['timestamp']} ]  UP: {up_prob:.2f} | DOWN: {down_prob:.2f}")

                        # Check trade filter conditions
                        squeeze_on = last_bar_data['squeeze_on']
                        adx = last_bar_data['adx']
                        vol_surge = last_bar_data['vol_surge']
                        
                        # --- LONG SIGNAL ---
                        if (up_prob > self.TRADE_PARAMS['ENTRY_CONF'] and 
                            adx > self.TRADE_PARAMS['ADX_THRESH'] and 
                            squeeze_on == 1 and 
                            vol_surge == 1):
                            
                            print("="*40)
                            print(f"🔥🔥🔥 LONG TRADE SIGNAL DETECTED 🔥🔥🔥")
                            print(f"  Confidence: {up_prob:.2f} > {self.TRADE_PARAMS['ENTRY_CONF']}")
                            print(f"  ADX: {adx:.1f} > {self.TRADE_PARAMS['ADX_THRESH']}")
                            print(f"  Squeeze: ON | Vol Surge: YES")
                            print(f"  Trade: Place LONG @ {last_bar_data['close']} with STOP at [PRICE - {self.TRADE_PARAMS['STOP_ATR_MULT']} * ATR]")
                            print("="*40)

                        # --- SHORT SIGNAL ---
                        elif (down_prob > self.TRADE_PARAMS['ENTRY_CONF'] and 
                              adx > self.TRADE_PARAMS['ADX_THRESH'] and 
                              squeeze_on == 1 and 
                              vol_surge == 1):
                            
                            print("="*40)
                            print(f"🥶🥶🥶 SHORT TRADE SIGNAL DETECTED 🥶🥶🥶")
                            print(f"  Confidence: {down_prob:.2f} > {self.TRADE_PARAMS['ENTRY_CONF']}")
                            print(f"  ADX: {adx:.1f} > {self.TRADE_PARAMS['ADX_THRESH']}")
                            print(f"  Squeeze: ON | Vol Surge: YES")
                            print(f"  Trade: Place SHORT @ {last_bar_data['close']} with STOP at [PRICE + {self.TRADE_PARAMS['STOP_ATR_MULT']} * ATR]")
                            print("="*40)

                except Exception as e:
                    print(f"❌ Error during AI prediction: {e}")
            
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
                        print(f"[Timer] Closing bar for {self.current_bar_time}")
                        await self._close_and_print_bar()

            except asyncio.CancelledError:
                print("Bar closer watcher stopping.")
                break
            except Exception as e:
                print(f"Error in bar_closer_watcher: {e}")
                await asyncio.sleep(1) # Don't spam errors


    async def handle_trade(self, trade):
        """        
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
                        "timestamp": bar_time.isoformat(),
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
                        self.current_bar = {"timestamp": bar_time.isoformat(), "open": price, "high": price, "low": price, "close": price, "volume": volume}
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
                        help='Contract to trade ie. CON.F.US.ES.Z25')
    parser.add_argument('--size', type=int, required=True,
                        help='Contract size')
    parser.add_argument('--username', type=str, required=True,
                        help='TopstepX username')
    parser.add_argument('--apikey', type=str, required=True,
                        help='TopstepX API key')
    
    # --- UPDATED ARGUMENTS ---
    parser.add_argument('--timeframe', type=int, choices=[1, 3, 5], default=5,
                        help='Timeframe in minutes (1, 3, 5). Default: 5')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the ONNX model file (e.g., SUPERTRADER_strategy_1_v2.onnx)')
    parser.add_argument('--scaler', type=str, required=True,
                        help='Path to the pickled scaler file (e.g., SUPERTRADER_scalers_v2.pkl)')
    
    args = parser.parse_args()
    
    # --- Check for ES contract (since model is ES specific) ---
    if 'ES' not in args.contract:
        print(f"⚠️  WARNING: You are running an ES-specific model, but your contract is {args.contract}.")
        print("The scaler will fail. Please trade an ES contract.")
        # return # You might want to exit here
    
    # Authenticate and get JWT token
    jwt_token = authenticate(args.username, args.apikey)

    if not jwt_token:
        print("\n❌ Cannot start bot without valid authentication")
        return
    
    print(f"\n🎫 Token received (expires in ~24 hours)")
    
    try:        
        # Pass model and scaler paths to the bot
        bot = RealTimeBot(
            jwt_token, 
            args.contract, 
            args.timeframe, 
            args.model, 
            args.scaler
        )        
        asyncio.run(bot.run())        
    except KeyboardInterrupt:
        print("\n👋 Bot stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()