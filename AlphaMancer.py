import ccxt
import numpy as np
import pandas as pd
import talib
from stable_baselines3 import PPO
import time

# Initialize Exchange (Binance as an example)
exchange = ccxt.binance({
    "apiKey": "YOUR_API_KEY",
    "secret": "YOUR_SECRET_KEY",
    "options": {"defaultType": "future"}  # For futures trading
})

# Fetch Market Data
def get_market_data(symbol="BTC/USDT", timeframe="5m", limit=100):
    candles = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    return df

# Compute Indicators
def compute_indicators(df):
    df["SMA"] = talib.SMA(df["close"], timeperiod=14)
    df["RSI"] = talib.RSI(df["close"], timeperiod=14)
    df["MACD"], df["MACD_signal"], _ = talib.MACD(df["close"], fastperiod=12, slowperiod=26, signalperiod=9)
    return df

# AI Decision Making
def ai_decision(model, df):
    state = df[["close", "SMA", "RSI", "MACD", "MACD_signal"]].values[-1].reshape(1, -1)
    action, _states = model.predict(state)
    return action

# Execute Trade
def execute_trade(action, symbol="BTC/USDT", amount=0.001):
    try:
        if action == 0:
            order = exchange.create_market_buy_order(symbol, amount)
            print("Bought", symbol, order)
        elif action == 1:
            order = exchange.create_market_sell_order(symbol, amount)
            print("Sold", symbol, order)
        else:
            print("Hold position")
    except Exception as e:
        print("Trade execution error:", str(e))

# Load AI Model (Pretrained PPO)
try:
    model = PPO.load("alphamancer_model.zip")
except:
    print("No pre-trained model found. Training required.")
    model = None

# Main Execution Loop
def run_bot():
    while True:
        try:
            df = get_market_data()
            df = compute_indicators(df)
            
            if model:
                action = ai_decision(model, df)
                execute_trade(action)
            else:
                print("No AI model loaded. Bot inactive.")
            
            time.sleep(300)  # Run every 5 minutes
        except Exception as e:
            print("Error in main loop:", str(e))

if __name__ == "__main__":
    run_bot()
