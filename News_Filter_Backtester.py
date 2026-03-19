import pandas as pd
import numpy as np
import joblib
import glob
import os
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
MODEL_PATH = 'Master_AI_Brain.pkl'
DATA_FOLDER = 'market_data_TEST'
NEWS_FILE = 'red_folder_news.csv'
STARTING_BALANCE = 100000   
RISK_PER_TRADE = 0.01
RR_RATIO = 3.0
TARGET_THRESHOLD = 0.60
NUM_SIMULATIONS = 1000
NEWS_WINDOW_MINUTES = 8  # 8 mins before and 8 mins after

print("Loading Master AI Brain and Historical Data...")

# --- 1. PARSE THE NEWS DATA ---
news_df = pd.read_csv(NEWS_FILE).dropna(subset=['Date', 'Time'])

def parse_news_time(row):
    date_str = str(row['Date'])
    time_str = str(row['Time'])
    
    # --- NEW LOGIC: Skip "All Day" or "Day X" events ---
    if "Day" in time_str or "All" in time_str:
        return pd.NaT
    
    # Remove the day of the week (e.g., "Wed Oct 1" -> "Oct 1")
    parts = date_str.split()
    if len(parts) >= 3 and parts[0][:3] in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']:
        date_str = ' '.join(parts[1:])
        
    # Inject the correct year based on your testing window
    if '202' not in date_str:
        if any(m in date_str for m in ['Oct', 'Nov', 'Dec']):
            date_str += ' 2025'
        else:
            date_str += ' 2026'
            
    return pd.to_datetime(f"{date_str} {time_str}")

# Apply the function and instantly drop the 'NaT' (Not a Time) rows we just skipped
news_df['Datetime'] = news_df.apply(parse_news_time, axis=1)
news_df = news_df.dropna(subset=['Datetime'])


# Convert to numpy array of datetime64 for lightning-fast math
news_times = news_df['Datetime'].values.astype('datetime64[ns]')

print(f"Loaded {len(news_times)} Red Folder events.")

# --- 2. LOAD & FORMAT TRADE DATA ---
model = joblib.load(MODEL_PATH)
expected_features = model.feature_names_in_

all_data = []
for file_path in glob.glob(os.path.join(DATA_FOLDER, '*.csv')):
    df = pd.read_csv(file_path)
    
    # Extract the features
    data = df['Message'].str.split(',', expand=True)
    if data.shape[1] > 11:
        data = data.iloc[:, :11]
    data.columns = ['dir', 'size_atr', 'overlap_pct', 'displacement', 'wick_ratio', 'dist_ema', 'pos_ema', 'session', 'outcome', 'bars_to_entry', 'bars_to_outcome']
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
        
    # CRITICAL: Keep the TradingView timestamp, convert to UTC-naive
    data['Trade_Time'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)
    all_data.append(data.dropna())

combined_df = pd.concat(all_data, ignore_index=True)
X = combined_df[expected_features]
probs = model.predict_proba(X)[:, 1]

# --- 3. APPLY THE AI THRESHOLD & NEWS FILTER ---
taken_mask = probs >= TARGET_THRESHOLD
taken_trades = combined_df[taken_mask].copy()

def is_safe_from_news(trade_time):
    # Calculate the exact distance in minutes to EVERY news event simultaneously
    trade_dt64 = np.datetime64(trade_time)
    diffs = np.abs(news_times - trade_dt64)
    min_diff = np.min(diffs)
    # If the closest news event is strictly greater than 8 mins away, it's safe
    return min_diff > np.timedelta64(NEWS_WINDOW_MINUTES, 'm')

# Run the trades through the bouncer
taken_trades['Safe'] = taken_trades['Trade_Time'].apply(is_safe_from_news)

safe_trades = taken_trades[taken_trades['Safe'] == True]
blocked_count = len(taken_trades) - len(safe_trades)
actual_outcomes = safe_trades['outcome'].values

print(f"\nAI signaled {len(taken_trades)} trades at 60% confidence.")
print(f"The News Filter blocked {blocked_count} trades that fell within the 16-minute window.")
print(f"Executing Monte Carlo on the remaining {len(safe_trades)} insulated trades...\n")

# --- 4. RUN MONTE CARLO ON INSULATED TRADES ---
fixed_risk_amount = STARTING_BALANCE * RISK_PER_TRADE
pnl_array = np.where(actual_outcomes == 1, fixed_risk_amount * RR_RATIO, -fixed_risk_amount)

all_drawdowns, all_max_streaks = [], []

for _ in range(NUM_SIMULATIONS):
    np.random.shuffle(pnl_array)
    equity = STARTING_BALANCE + np.cumsum(pnl_array)
    equity_curve = np.insert(equity, 0, STARTING_BALANCE)
    
    peak = np.maximum.accumulate(equity_curve)
    drawdowns = (peak - equity_curve) / peak * 100
    all_drawdowns.append(np.max(drawdowns))
    
    is_loss = (pnl_array < 0)
    streak = max_streak = 0
    for loss in is_loss:
        if loss:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    all_max_streaks.append(max_streak)

# --- 5. THE RESULTS ---
print("==================================================")
print("     INSULATED MONTE CARLO RESULTS (NEWS AVOIDED)")
print("==================================================")
print(f"New 99th Percentile Drawdown: {np.percentile(all_drawdowns, 99):.2f}%")
print(f"New Absolute WORST Drawdown:  {np.max(all_drawdowns):.2f}%")
print(f"New WORST Losing Streak:      {np.max(all_max_streaks)} trades in a row")
print("==================================================")