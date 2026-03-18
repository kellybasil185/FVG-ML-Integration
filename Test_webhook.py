import requests
import json

# Your live Ngrok Webhook URL (Make sure /webhook is at the end!)
WEBHOOK_URL = "https://1b60-102-176-94-123.ngrok-free.app/webhook"

# We are building a "God Tier" fake setup so the AI is guaranteed to score it >70%
fake_trade_data = {
    "ticker": "EURUSD",
    "dir": 1,               # 1 for Buy, 0 for Sell
    "size_atr": 2.5,        # Massive FVG
    "overlap_pct": 100,     # Perfect overlap
    "displacement": 4.5,    # Huge momentum push
    "wick_ratio": 0.02,     # Almost zero wick, pure institutional buying
    "dist_ema": 850,        # Great distance to the 200 EMA
    "pos_ema": 1,           # Above the EMA
    "session": 2,           # London Session
    "entry": 1.08500,
    "sl": 1.08450,
    "tp": 1.08650           # 1:3 RR Target
}

print(f"Firing test payload to {WEBHOOK_URL}...")

try:
    # Send the POST request just like TradingView will
    response = requests.post(WEBHOOK_URL, json=fake_trade_data)
    
    print(f"Server Response Code: {response.status_code}")
    print(f"Server Message: {response.text}")
    
except Exception as e:
    print(f"Failed to connect: {e}")