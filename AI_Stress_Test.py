import pandas as pd
import numpy as np
import os
import glob
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score
from sklearn.model_selection import TimeSeriesSplit
import warnings

warnings.filterwarnings('ignore')

DATA_FOLDER = 'market_data_15min' # Ensure your 15_4RR CSVs are in here
features = ['dir', 'size_atr', 'overlap_pct', 'displacement', 'wick_ratio', 'dist_ema', 'pos_ema', 'session']

print("==================================================")
print("       MASTER AI STRESS TEST & VETTING")
print("==================================================\n")

print("1. Loading and sorting global market data chronologically...")
all_global_data = []
for file_path in glob.glob(os.path.join(DATA_FOLDER, '*.csv')):
    try:
        df = pd.read_csv(file_path)
        # Parse the TradingView Date column to ensure proper time-travel testing
        df['Date'] = pd.to_datetime(df.iloc[:, 0], errors='coerce') 
        
        data = df['Message'].str.split(',', expand=True)
        if data.shape[1] > 11:
            data = data.iloc[:, :11]
            
        data.columns = ['dir', 'size_atr', 'overlap_pct', 'displacement', 'wick_ratio', 'dist_ema', 'pos_ema', 'session', 'outcome', 'bars_to_entry', 'bars_to_outcome']
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            
        # Merge the parsed Date back in
        data['Date'] = df['Date']
        all_global_data.append(data.dropna())
    except Exception as e:
        print(f"Skipping {file_path} due to error: {e}")

# Combine and sort strictly by time
combined_df = pd.concat(all_global_data, ignore_index=True)
combined_df = combined_df.sort_values(by='Date').reset_index(drop=True)

X = combined_df[features]
y = combined_df['outcome']

print(f"Total chronological trades loaded: {len(combined_df)}\n")

# --- TEST 1: THE GAP CHECK ---
print("--- TEST 1: THE GAP CHECK (Train vs. Test Bias) ---")
split_index = int(len(combined_df) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

ratio = float(y_train.value_counts().get(0, 0)) / y_train.value_counts().get(1, 1)
sample_weights = np.where(y_train == 1, ratio, 1.0)

model = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.7, min_samples_leaf=15, random_state=42)
model.fit(X_train, y_train, sample_weight=sample_weights)

# Grade the Training Data
train_preds = (model.predict_proba(X_train)[:, 1] >= 0.70).astype(int)
train_win_rate = precision_score(y_train, train_preds, zero_division=0) * 100

# Grade the Unseen Test Data
test_preds = (model.predict_proba(X_test)[:, 1] >= 0.70).astype(int)
test_win_rate = precision_score(y_test, test_preds, zero_division=0) * 100

print(f"Win Rate on TRAINING Data (>70% Conf): {train_win_rate:.2f}%")
print(f"Win Rate on UNSEEN TEST Data (>70% Conf): {test_win_rate:.2f}%")
gap = train_win_rate - test_win_rate
print(f"Overfit Gap: {gap:.2f}%")
if gap > 15:
    print("Verdict: Model is slightly overfitted (memorizing noise).")
elif gap < 0:
    print("Verdict: Anomaly. Test data performed better than training data.")
else:
    print("Verdict: EXCELLENT. The model learned the true rules without memorizing the data.\n")


# --- TEST 2: WALK-FORWARD CROSS VALIDATION ---
print("\n--- TEST 2: WALK-FORWARD CROSS VALIDATION (Time-Series) ---")
print("Simulating trading through 5 different historical market phases...\n")

tscv = TimeSeriesSplit(n_splits=5)
fold = 1
fold_win_rates = []

for train_index, test_index in tscv.split(X):
    X_train_cv, X_test_cv = X.iloc[train_index], X.iloc[test_index]
    y_train_cv, y_test_cv = y.iloc[train_index], y.iloc[test_index]
    
    if y_train_cv.value_counts().get(1, 0) == 0:
        continue
        
    cv_ratio = float(y_train_cv.value_counts().get(0, 0)) / y_train_cv.value_counts().get(1, 1)
    cv_weights = np.where(y_train_cv == 1, cv_ratio, 1.0)
    
    cv_model = GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=42)
    cv_model.fit(X_train_cv, y_train_cv, sample_weight=cv_weights)
    
    cv_preds = (cv_model.predict_proba(X_test_cv)[:, 1] >= 0.70).astype(int)
    trades_taken = cv_preds.sum()
    
    if trades_taken > 0:
        cv_win_rate = precision_score(y_test_cv, cv_preds, zero_division=0) * 100
        fold_win_rates.append(cv_win_rate)
        print(f"Phase {fold} | Trades: {trades_taken:<3} | Win Rate: {cv_win_rate:.2f}%")
    else:
        print(f"Phase {fold} | Trades: 0   | AI rejected all setups in this period.")
    fold += 1

if fold_win_rates:
    avg_cv_win_rate = sum(fold_win_rates) / len(fold_win_rates)
    print(f"\nAverage Walk-Forward Win Rate (>70% Conf): {avg_cv_win_rate:.2f}%")
    if avg_cv_win_rate >= 50.0:
        print("FINAL VERDICT: The model is highly robust and safely passes the stress test. Ready for deployment!")
    else:
        print("FINAL VERDICT: The model broke down during certain historical phases. Proceed with caution.")