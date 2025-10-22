#!/usr/bin/env python3
"""
Real-Time Trading Bot with SignalR - Aggregates ticks and runs AI model.

This version uses command-line arguments for trading parameters, allowing
flexibility to run different strategies (ES, NQ, YM, RTY) without code changes.

Requirements:
    pip install onnxruntime pandas pandas-ta signalrcore requests numpy scikit-learn python-dateutil
"""

import numpy as np
import asyncio
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

MARKET_HUB = "https://rtc.topstepx.com/hubs/market"
BASE_URL = "https://api.topstepx.com/api"

CONTRACTS = {
    "CON.F.US.ENQ.Z25" : 'NQ',
    "CON.F.US.EP.Z25" : 'ES',
    "CON.F.US.YM.Z25" : 'YM',
    "CON.F.US.RTY.Z25" : 'RTY'
}

# =========================================================
# AUTHENTICATION
# =========================================================
def authenticate(username, api_key):
    """Authenticates and returns a JWT token."""
    auth_url = f"{BASE_URL}/Auth/loginKey"
    payload = {"userName": username, "apiKey": api_key}
    try:
        print("🔐 Authenticating...")
        response = requests.post(auth_url, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get('success') and data.get('token'):
            print("✅ Authentication successful!")
            return data['token']
        else:
            print(f"❌ Authentication failed: {data.get('errorMessage', 'Unknown error')}")
            return None
    except Exception as e:
        print(f"❌ Authentication error: {e}")
        return None

# =========================================================
# UTILITY FUNCTION
# =========================================================
def parse_contract_symbol(contract_id):
    """Extracts the base symbol (e.g., 'ES', 'RTY') from the full contract ID."""
    return CONTRACTS[contract_id]    

# =========================================================
# REAL-TIME TRADING BOT CLASS
# =========================================================
class RealTimeBot:    
    def __init__(self, token,account, contract, size, timeframe_minutes, model_path, scaler_path,
                 entry_conf, adx_thresh, stop_atr, target_atr, enable_trailing_stop=False):
        self.hub_url = f"{MARKET_HUB}?access_token={token}"
        self.account = account
        self.contract = contract
        self.size = size
        self.timeframe_minutes = int(timeframe_minutes)
        self.client = SignalRClient(self.hub_url)
        self.token = token
        self.enable_trailing_stop = enable_trailing_stop
                
        self.current_bar = {}
        self.current_bar_time = None
        self.bar_lock = asyncio.Lock()
        self.closer_task = None
        
        # --- State Management ---
        self.in_position = False
        self.position_type = None
        self.entry_price = None
        self.stop_loss = None
        self.profit_target = None
        self.stop_orderId = None
        self.limit_orderId = None        
        
        # --- AI & Data Properties ---
        self.num_historical_candles_needed = 60
        self.historical_bars = deque(maxlen=60)
        self.model_path = model_path
        self.scaler_path = scaler_path
        
        self.ort_session = None
        self.scalers = None
        self.active_scaler = None
        self.contract_symbol = parse_contract_symbol(self.contract)
        
        self.feature_cols = [
            'compression_level', 'squeeze_duration', 'bb_expanding',
            'atr_expanding', 'price_in_range', 'rsi',
            'compressed_momentum', 'vol_surge', 'body_strength',
        ]
                
        self.entry_conf = entry_conf
        self.adx_thresh = adx_thresh
        self.stop_atr_mult = stop_atr
        self.target_atr_mult = target_atr        
        
        print(f"🤖 Bot initialized for {self.contract} ({self.contract_symbol}) on {self.timeframe_minutes}-min timeframe.")
        print(f"📈 Using Trade Params: Entry={self.entry_conf}, ADX={self.adx_thresh}, Stop={self.stop_atr_mult} ATR, Target={self.target_atr_mult} ATR")
        
        self.load_model_and_scalers()

        # Register handlers
        self.client.on_open(self.on_open)
        self.client.on_close(self.on_close)
        self.client.on_error(self.on_error)
        self.client.on("GatewayTrade", self.process_tick)

    def load_model_and_scalers(self):
        """Loads the ONNX model and the correct scaler for the contract symbol."""
        try:
            if not os.path.exists(self.model_path): raise FileNotFoundError(f"Model file not found: {self.model_path}")
            if not os.path.exists(self.scaler_path): raise FileNotFoundError(f"Scaler file not found: {self.scaler_path}")
            
            self.ort_session = onnxruntime.InferenceSession(self.model_path)
            print(f"✅ Successfully loaded ONNX model: {os.path.basename(self.model_path)}")

            with open(self.scaler_path, 'rb') as f:
                self.scalers = pickle.load(f)
            
            # --- MODIFIED: Use parsed contract_symbol to get scaler ---
            if self.scalers and self.contract_symbol in self.scalers:
                self.active_scaler = self.scalers[self.contract_symbol]
                print(f"✅ Successfully loaded and extracted '{self.contract_symbol}' scaler from: {os.path.basename(self.scaler_path)}")
            else:
                # <-- MODIFIED: More specific error message -->
                available_keys = list(self.scalers.keys()) if self.scalers else "None"
                raise ValueError(f"'{self.contract_symbol}' scaler not found in {os.path.basename(self.scaler_path)}. Available scalers: {available_keys}")

        except Exception as e:
            print(f"❌ CRITICAL ERROR loading model/scalers: {e}")
            raise e

    # --- add_ai_features remains the same ---
    def add_ai_features(self, df):
        """Calculates all necessary AI features on the historical dataframe."""
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        bbands = ta.bbands(df['close'], length=20, std=2)
        # Handle potential variations in pandas_ta column naming
        bb_upper_col = next((col for col in bbands.columns if col.startswith('BBU_')), 'BBU_20_2.0_2.0')
        bb_lower_col = next((col for col in bbands.columns if col.startswith('BBL_')), 'BBL_20_2.0_2.0')
        bb_mid_col = next((col for col in bbands.columns if col.startswith('BBM_')), 'BBM_20_2.0_2.0')
        df['bb_upper'] = bbands[bb_upper_col]
        df['bb_lower'] = bbands[bb_lower_col]
        df['bb_mid'] = bbands[bb_mid_col]

        keltner = ta.kc(df['high'], df['low'], df['close'], length=20, mamode='ema', atr_length=10, multiplier=2)
        kc_upper_col = next((col for col in keltner.columns if col.startswith('KCUe_')), 'KCUe_20_2')
        kc_lower_col = next((col for col in keltner.columns if col.startswith('KCLe_')), 'KCLe_20_2')
        df['kc_upper'] = keltner[kc_upper_col]
        df['kc_lower'] = keltner[kc_lower_col]

        df['rsi'] = ta.rsi(df['close'], length=14)
        df['roc'] = ta.roc(df['close'], length=10)
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
        adx_col = next((col for col in adx_df.columns if col.startswith('ADX_')), 'ADX_14')
        df['adx'] = adx_df[adx_col]

        df['vol_ma'] = df['volume'].rolling(20).mean()
        df['squeeze_on'] = ((df['bb_lower'] > df['kc_lower']) & (df['bb_upper'] < df['kc_upper'])).astype(float)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'].replace(0, np.nan) # Avoid division by zero
        df['bb_width_pct'] = df['bb_width'] / df['bb_width'].rolling(50).mean()
        df['compression_level'] = 1 - df['bb_width_pct'].clip(0, 2)
        df['squeeze_duration'] = df['squeeze_on'].groupby((df['squeeze_on'] != df['squeeze_on'].shift()).cumsum()).cumsum()
        df['bb_expanding'] = (df['bb_width'].diff(3) > 0).astype(float)
        df['atr_expanding'] = (df['atr'].pct_change(3) > 0.03).astype(float)
        df['price_in_range'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']).replace(0, np.nan) # Avoid division by zero
        df['compressed_momentum'] = df['roc'] * df['squeeze_on']
        df['vol_ratio'] = df['volume'] / df['vol_ma'].replace(0, 1)
        df['vol_surge'] = (df['vol_ratio'] > 1.3).astype(float)
        df['body'] = (df['close'] - df['open']) / df['atr'].replace(0, np.nan) # Avoid division by zero
        df['body_strength'] = abs(df['body'])
        
        # Fill NaNs more robustly
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(method='ffill', inplace=True)
        df.fillna(0, inplace=True)
        return df
    
    def get_ai_prediction(self):
        """Runs the ONNX model to get a prediction."""
        if len(self.historical_bars) < self.num_historical_candles_needed:
            print(f"⏳ Warming up... {len(self.historical_bars)}/{self.num_historical_candles_needed} bars.")
            return None, None, None

        df = pd.DataFrame(list(self.historical_bars))
        df_with_features = self.add_ai_features(df)
        last_60_bars_features = df_with_features.tail(self.num_historical_candles_needed)
        last_bar_data = df_with_features.iloc[-1]
        
        if last_60_bars_features[self.feature_cols].isnull().values.any():
            print("⏳ Indicators still warming up (NaNs found). Skipping prediction.")
            return None, None, None

        features_array = last_60_bars_features[self.feature_cols].values
        # Ensure scaler is ready
        if not self.active_scaler:
             print("❌ Scaler not loaded. Cannot make prediction.")
             return None, None, None
        scaled_features = self.active_scaler.transform(features_array)
        model_input = np.array([scaled_features], dtype=np.float32)

        ort_inputs = {self.ort_session.get_inputs()[0].name: model_input}
        ort_outs = self.ort_session.run(None, ort_inputs)
        
        logits = ort_outs[0][0]
        probs = np.exp(logits) / np.sum(np.exp(logits))
        return probs[0], probs[1], last_bar_data
    
    async def fetch_historical_data(self):
        """Fetches recent bars to prime the historical data deque."""
        historical_url = f"{BASE_URL}/History/retrieveBars"        
        end_time_dt = datetime.now(timezone.utc).replace(microsecond=0)
        # Fetch a bit more history (e.g., 3 days) to ensure enough warmup for indicators like 50-period rolling mean
        start_time_dt = end_time_dt - relativedelta(days=3)
        payload = {            
            "contractId": self.contract, "live": False,
            "startTime": start_time_dt.isoformat().replace('+00:00', 'Z'),
            "endTime": end_time_dt.isoformat().replace('+00:00', 'Z'),
            "unit": 2, "unitNumber": self.timeframe_minutes,
            "limit": self.historical_bars.maxlen, "includePartialBar": False
        }
        headers = {'Authorization': f'Bearer {self.token}'}
        try:            
            response = requests.post(historical_url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            bars_fetched = 0
            for bar in response.json().get('bars', []):                
                self.historical_bars.append({
                    "timestamp": datetime.fromisoformat(bar['t']).isoformat(),
                    "open": bar['o'], "high": bar['h'], "low": bar['l'],
                    "close": bar['c'], "volume": bar['v']
                })
                bars_fetched += 1
            print(f"✅ Successfully pre-filled {bars_fetched} historical bars.")
            if bars_fetched < self.num_historical_candles_needed:
                 print(f"⚠️ Warning: Fetched fewer bars ({bars_fetched}) than needed ({self.num_historical_candles_needed}) for full AI warmup.")
        except Exception as e:
            print(f"❌ Could not fetch historical data: {e}.")
    
    async def run(self):
        """Starts the bot."""
        await self.fetch_historical_data()
        print("🚀 Starting bot connection...")                
        self.closer_task = asyncio.create_task(self.bar_closer_watcher())        
        await self.client.run()
    
    async def on_open(self):
        print("✅ Connected to market hub")
        try: # Added error handling
            await self.client.send("SubscribeContractTrades", [self.contract])
            print(f"✅ Subscription successful for {self.contract}")
        except Exception as e:
            print(f"❌ Subscription error: {e}")

    
    async def on_close(self):
        print('🔌 Disconnected from the server')
        if self.closer_task: self.closer_task.cancel()
    
    async def on_error(self, message):
        print(f"❌ SignalR Error: {message}")
    
    async def process_tick(self, data):
        """Processes incoming tick data."""
        try:
            # Handle list format: ['CONTRACT', [ {...}, {...} ]]
            if isinstance(data, list) and len(data) >= 2 and isinstance(data[1], list):
                trades = data[1]
            # Handle single dict format: { ... }
            elif isinstance(data, dict):
                trades = [data]
            else:
                # print(f"❓ Unexpected tick data format: {data}") # Can be noisy
                trades = []

            for trade in trades:
                await self.handle_trade(trade)
        except Exception as e:
            print(f"❌ process_tick error: {e} | Data: {data}")

    # --- _get_bar_time remains the same ---    
    def _get_bar_time(self, ts):
        """Floors a timestamp to the correct bar interval."""
        if ts.tzinfo is None: ts = ts.replace(tzinfo=timezone.utc)
        else: ts = ts.astimezone(timezone.utc)
        bar_time = ts.replace(second=0, microsecond=0)
        if self.timeframe_minutes > 1:
            bar_time = bar_time.replace(minute=(bar_time.minute // self.timeframe_minutes) * self.timeframe_minutes)
        return bar_time
    
    async def _place_order(self, side, type = 2, limitPrice = None, stopPrice=None, trailPrice=None):
        """enter long/short position"""
        order_url = f"{BASE_URL}/Order/place"
        payload = {            
            "accountId": self.account, 
            "contractId": self.contract,
            "type": type, # Market order
            "side": side, # 0 - Bid (buy), 1 = Ask (sell)
            "limitPrice": limitPrice,
            "stopPrice": stopPrice,
            "trailPrice": trailPrice,
            "size": self.size # Size of the order
        }
        headers = {'Authorization': f'Bearer {self.token}'}
        try:            
            response = requests.post(order_url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()  
            if data.get('success') and data.get('orderId'):
                return data.get("orderId")
            else: return None
        except Exception as e:
            print(f"❌ Could not place order: {e}.")
        

    async def _close_and_print_bar(self):
        """On bar close, appends data and checks for a NEW trade entry."""
        if not self.current_bar: return
        
        #print(f"Bar Close: {self.current_bar}")            
        self.historical_bars.append(self.current_bar.copy())
        
        # --- ENTRY LOGIC (only if not already in a position) ---
        if not self.in_position and self.ort_session and self.active_scaler:
            try:
                down_prob, up_prob, last_bar = self.get_ai_prediction()
                
                # Check if prediction was successful
                if down_prob is not None and last_bar is not None:
                    print(f"🧠 AI PREDICTION [ {last_bar['timestamp']} ]  UP: {up_prob:.2f} | DOWN: {down_prob:.2f}")
                                        
                    is_long_signal = (up_prob > self.entry_conf and 
                                      last_bar['adx'] > self.adx_thresh and 
                                      last_bar['squeeze_on'] == 1 and 
                                      last_bar['vol_surge'] == 1)
                    
                    is_short_signal = (down_prob > self.entry_conf and 
                                       last_bar['adx'] > self.adx_thresh and 
                                       last_bar['squeeze_on'] == 1 and 
                                       last_bar['vol_surge'] == 1)

                    # --- Entry logic using instance attributes ---
                    if is_long_signal:
                        self.in_position = True
                        self.position_type = 'LONG'
                        self.entry_price = last_bar['close']
                        self.stop_loss = self.entry_price - (last_bar['atr'] * self.stop_atr_mult)
                        self.profit_target = self.entry_price + (last_bar['atr'] * self.target_atr_mult)
                        print("="*40, f"\n🔥🔥🔥 ENTERING LONG @ {self.entry_price:.2f} 🔥🔥🔥", f"\n  SL: {self.stop_loss:.2f} | PT: {self.profit_target:.2f}", "\n"+"="*40)
                        await self._place_order(0)
                        self.limit_orderId = await self._place_order(1, type=1, limitPrice=self.profit_target) # SELL LIMIT order
                        if self.enable_trailing_stop:
                            self.stop_orderId = await self._place_order(1, type=5, trailPrice=self.stop_loss) # SELL TRAIL STOP order
                        else:
                            self.stop_orderId = await self._place_order(1, type=4, stopPrice=self.stop_loss) # SELL STOP order                        

                    elif is_short_signal:
                        self.in_position = True
                        self.position_type = 'SHORT'
                        self.entry_price = last_bar['close']
                        self.stop_loss = self.entry_price + (last_bar['atr'] * self.stop_atr_mult)
                        self.profit_target = self.entry_price - (last_bar['atr'] * self.target_atr_mult)
                        print("="*40, f"\n🥶🥶🥶 ENTERING SHORT @ {self.entry_price:.2f} 🥶🥶🥶", f"\n  SL: {self.stop_loss:.2f} | PT: {self.profit_target:.2f}", "\n"+"="*40)
                        await self._place_order(1)
                        self.limit_orderId = await self._place_order(0, type=1, limitPrice=self.profit_target) # BUY LIMIT order
                        if self.enable_trailing_stop:
                            self.stop_orderId = await self._place_order(0, type=5, trailPrice=self.stop_loss) # BUY TRAIL STOP order
                        else:
                            self.stop_orderId = await self._place_order(0, type=4, stopPrice=self.stop_loss) # BUY STOP order
            except Exception as e:
                print(f"❌ Error during AI prediction/Entry Logic: {e}")
        
        # Reset bar state
        self.current_bar, self.current_bar_time = {}, None
    
    async def bar_closer_watcher(self):
        """Background task to watch the clock and force-close bars."""
        print("⏳ Bar closer watcher started...")
        while True:
            try:
                if not self.current_bar_time:
                    await asyncio.sleep(0.1)
                    continue
                next_bar_time = self.current_bar_time + timedelta(minutes=self.timeframe_minutes)
                sleep_duration = (next_bar_time - datetime.now(timezone.utc)).total_seconds()
                if sleep_duration > 0: await asyncio.sleep(sleep_duration + 0.05)

                async with self.bar_lock:
                    if self.current_bar and self.current_bar_time < next_bar_time:
                        # print(f"[Timer] Closing bar for {self.current_bar_time}") # Can be noisy
                        await self._close_and_print_bar()
            except asyncio.CancelledError: 
                print("Bar closer watcher stopping.")
                break
            except Exception as e: 
                print(f"Error in bar_closer_watcher: {e}")
                await asyncio.sleep(1) # Prevent rapid error loops
                
    async def _exit_position(self):
        """exit long/short position"""
        exit_position_url = f"{BASE_URL}/Position/closeContract"
        payload = {            
            "accountId": self.account, 
            "contractId": self.contract, 
        }
        headers = {'Authorization': f'Bearer {self.token}'}
        try:            
            response = requests.post(exit_position_url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()        
        except Exception as e:
            print(f"❌ Could not exit position: {e}.")
            self.in_position = False

    async def _cancel_order(self, orderId):
        """exit long/short position"""
        cancel_order_url = f"{BASE_URL}/Order/cancel"
        payload = {            
            "accountId": self.account,
            "orderId": orderId, 
        }
        headers = {'Authorization': f'Bearer {self.token}'}
        try:            
            response = requests.post(cancel_order_url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()        
        except Exception as e:
            print(f"❌ Could not cancel order: {e}.")
            self.in_position = False

    async def handle_trade(self, trade):
        """Aggregates ticks into bars and checks for exits on every tick."""
        try:
            ts = datetime.fromisoformat(trade.get("timestamp"))
            price = trade.get("price")
            volume = trade.get("volume", 0)

            if price is None: return
            
            if self.in_position:
                exit_price, exit_reason = None, None                
                if self.stop_loss is None or self.profit_target is None:
                     print("⚠️ Exit check skipped: stop_loss or profit_target not set.")
                elif self.position_type == 'LONG':
                    if price <= self.stop_loss: exit_price, exit_reason = self.stop_loss, 'STOP_LOSS'
                    elif price >= self.profit_target: exit_price, exit_reason = self.profit_target, 'PROFIT_TARGET'
                elif self.position_type == 'SHORT':
                    if price >= self.stop_loss: exit_price, exit_reason = self.stop_loss, 'STOP_LOSS'
                    elif price <= self.profit_target: exit_price, exit_reason = self.profit_target, 'PROFIT_TARGET'            
                
                if exit_reason:
                    pnl = (exit_price - self.entry_price) if self.position_type == 'LONG' else (self.entry_price - exit_price)
                    print("="*40, f"\n🛑🛑🛑 EXIT {self.position_type} @ {exit_price:.2f} ({exit_reason}) 🛑🛑🛑", f"\n  Entry: {self.entry_price:.2f} | PnL Points: {pnl:.2f}", "\n"+"="*40)                    
                    if exit_reason == 'STOP_LOSS' and self.stop_orderId:
                        self._cancel_order(self.stop_orderId)
                    elif exit_reason == 'PROFIT_TARGET' and self.limit_orderId:
                        self._cancel_order(self.limit_orderId)
                    self.in_position, self.position_type, self.entry_price, self.stop_loss, self.profit_target = False, None, None, None, None

            # --- Bar Aggregation Logic ---
            bar_time = self._get_bar_time(ts)
            async with self.bar_lock:
                if bar_time != self.current_bar_time:
                    if self.current_bar: 
                        # print(f"[Tick] Closing bar for {self.current_bar_time}") # Can be noisy
                        await self._close_and_print_bar()
                    self.current_bar_time = bar_time
                    self.current_bar = {"timestamp": bar_time.isoformat(), "open": price, "high": price, "low": price, "close": price, "volume": volume}                
                elif self.current_bar: 
                    self.current_bar["high"] = max(self.current_bar.get("high", price), price)
                    self.current_bar["low"] = min(self.current_bar.get("low", price), price)
                    self.current_bar["close"] = price
                    self.current_bar["volume"] = self.current_bar.get("volume", 0) + volume
                else:                      
                     self.current_bar_time = bar_time
                     self.current_bar = {"timestamp": bar_time.isoformat(), "open": price, "high": price, "low": price, "close": price, "volume": volume}

        except Exception as e:
            print(f"❌ handle_trade error: {e} | Trade: {trade}")

# =========================================================
# MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description='Real-Time AI Futures Trading Bot',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Example Usage (RTY Strategy from Backtest #10):
  python %(prog)s --account ACC123 --contract CON.F.US.RTY.Z25 --size 1 \\
                  --username USER --apikey KEY --timeframe 5 \\
                  --model models/model1.onnx \\
                  --scaler models/strategy1.pkl \\
                  --entry_conf 0.55 --adx_thresh 25 --stop_atr 2.0 --target_atr 3.0
"""
    )
    # --- Existing Args ---
    parser.add_argument('--account', type=str, required=True, help='TopstepX account ID')
    parser.add_argument('--contract', type=str, required=True, help='Full contract ID (e.g., CON.F.US.RTY.Z25)')
    parser.add_argument('--size', type=int, required=True, help='Trade size (number of contracts)')
    parser.add_argument('--username', type=str, required=True, help='TopstepX username')
    parser.add_argument('--apikey', type=str, required=True, help='TopstepX API key')
    parser.add_argument('--timeframe', type=int, choices=[1, 3, 5], default=5, help='Bar timeframe in minutes (default: 5)')
    parser.add_argument('--enable_trailing_stop', type=bool, default=False, help='Enable trailing stop vs stop order')
        
    parser.add_argument('--model', type=str, default="models/model1.onnx", help='Path to the ONNX model file (.onnx)')
    parser.add_argument('--scaler', type=str, default="models/strategy1.pkl", help='Path to the pickled scaler file (.pkl)')
    parser.add_argument('--entry_conf', type=float, default=0.55, help='Min AI confidence to enter (default: 0.55)')
    parser.add_argument('--adx_thresh', type=int, default=25, help='Min ADX value to enter (default: 25)')
    parser.add_argument('--stop_atr', type=float, default=2.0, help='Stop loss multiplier (x ATR) (default: 2.0)')
    parser.add_argument('--target_atr', type=float, default=3.0, help='Profit target multiplier (x ATR) (default: 3.0)')    
    
    args = parser.parse_args()
    
    jwt_token = authenticate(args.username, args.apikey)
    if not jwt_token: return
    
    print(f"\n🎫 Token received. Bot starting...")
    try:
        # --- MODIFIED: Pass all parameters to the bot ---
        bot = RealTimeBot(
            token=jwt_token,
            account=args.account,
            contract=args.contract,
            size=args.size,
            timeframe_minutes=args.timeframe,
            model_path=args.model,
            scaler_path=args.scaler,
            entry_conf=args.entry_conf,
            adx_thresh=args.adx_thresh,
            stop_atr=args.stop_atr,
            target_atr=args.target_atr,
            enable_trailing_stop=args.enable_trailing_stop
        )
        asyncio.run(bot.run())
    except KeyboardInterrupt: print("\n👋 Bot stopped by user.")
    except Exception as e: print(f"\n❌ A critical error occurred: {e}")

if __name__ == "__main__":
    main()

