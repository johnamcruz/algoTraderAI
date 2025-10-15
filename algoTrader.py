#!/usr/bin/env python3
"""
Real-Time Trading Bot with SignalR - Listens to live bar updates

Requirements:
    pip install onnxruntime pandas pandas-ta signalrcore requests joblib

Usage:
    python algoTrader.py --ticker NQ --username YOUR_USERNAME --apikey YOUR_API_KEY
"""

#import onnxruntime as ort
import numpy as np
#import pandas as pd
#import pandas_ta as ta
#import joblib
import json
import argparse
import requests
from datetime import datetime, timedelta
#from signalrcore.hub_connection_builder import HubConnectionBuilder
import threading
import time
import warnings
warnings.filterwarnings('ignore')

# =========================================================
# AUTHENTICATION
# =========================================================

def authenticate(username, api_key):
    """
    Authenticate with TopstepX and get JWT token
    
    Returns: JWT token string or None if failed
    """
    auth_url = "https://api.topstepx.com/api/Auth/loginKey"
    
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


if __name__ == "__main__":
    main()