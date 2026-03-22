from flask import Flask, request, jsonify
import pandas as pd
import joblib
import requests
import os

# --- CLOUD CONFIGURATION ---
# These will be hidden inside Railway's settings!
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')
CONFIDENCE_THRESHOLD = 0.60  

app = Flask(__name__)

# --- THE HOLDING CELL ---
# This dictionary stores approved trades waiting for MT5 to fetch them
approved_trades = {}

print("Loading Master AI Brain...")
try:
    model = joblib.load('Master_AI_Brain.pkl')
    print("AI Brain Loaded Successfully!")
except Exception as e:
    print(f"Error loading AI model: {e}")

def send_telegram_alert(ticker, direction, entry, sl, tp, prob):
    dir_emoji = "🟢 BUY" if direction == 1 else "🔴 SELL"
    msg = (
        f"🚨 *AI SNIPER ALERT: {ticker}* 🚨\n\n"
        f"**Direction:** {dir_emoji}\n"
        f"**Entry:** {entry}\n"
        f"**Stop Loss:** {sl}\n"
        f"**Take Profit:** {tp}\n\n"
        f"🧠 *AI Confidence:* {prob:.1f}%"
    )
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': msg, 'parse_mode': 'Markdown'}
    requests.post(url, json=payload)

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        data = request.json
        
        # 1. We know the model wants these exact features.
        features = ['dir', 'size_atr', 'overlap_pct', 'displacement', 'wick_ratio', 'dist_ema', 'pos_ema', 'session']
        
        # 2. Build a dictionary mapping the incoming JSON to the exact feature names
        input_dict = {f: [data[f]] for f in features}
        
        # 3. Create the DataFrame
        df = pd.DataFrame(input_dict)
        
        # 4. CRITICAL FIX: Reorder the DataFrame columns to match the model's exact expected order
        expected_features = model.feature_names_in_
        df = df[expected_features]
        
        # 5. Ask the AI to grade the setup
        probability = model.predict_proba(df)[0][1] * 100 
        print(f"[{data['ticker']}] AI Graded: {probability:.2f}%")
        
        if probability >= (CONFIDENCE_THRESHOLD * 100):
            # Action A: Send Telegram Alert
            send_telegram_alert(data['ticker'], data['dir'], data['entry'], data['sl'], data['tp'], probability)
            
            # Action B: Save to MT5 Holding Cell
            approved_trades[data['ticker']] = data
            print(f"⚙️ Trade saved to holding cell for MT5: {data['ticker']}")
            
        return jsonify({'status': 'success'}), 200
    except Exception as e:
        print(f"Webhook Error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 400

# --- THE METATRADER 5 POLLING ENDPOINT ---
@app.route('/get_trade', methods=['GET'])
def send_to_mt5():
    # MT5 will request its specific pair (e.g., /get_trade?pair=USDJPY)
    requested_pair = request.args.get('pair')
    
    if requested_pair in approved_trades:
        # .pop() grabs the trade data AND deletes it from memory so it executes only once
        trade_data = approved_trades.pop(requested_pair)
        print(f"✅ MT5 fetched and cleared trade for {requested_pair}")
        return jsonify(trade_data), 200
        
    # If nothing is in the holding cell, tell MT5 to keep waiting
    return jsonify({"status": "no_trade"}), 200

if __name__ == '__main__':
    # Railway assigns a dynamic port, so we must catch it
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)