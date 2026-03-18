import pandas as pd
import numpy as np
import os
import glob
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score
import warnings

# Suppress warnings for clean console output
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
DATA_FOLDER = 'market_data_15min' # Folder where you put your CSVs
SUMMARY_FILE = 'Strategy_Scorecard_15min.txt'

columns = ['dir', 'size_atr', 'overlap_pct', 'displacement', 'wick_ratio', 'dist_ema', 'pos_ema', 'session', 'outcome', 'bars_to_entry', 'bars_to_outcome']
features = ['dir', 'size_atr', 'overlap_pct', 'displacement', 'wick_ratio', 'dist_ema', 'pos_ema', 'session']

# Find all CSVs in the folder
csv_files = glob.glob(os.path.join(DATA_FOLDER, '*.csv'))

if not csv_files:
    print(f"No CSV files found in the '{DATA_FOLDER}' folder. Please add them and try again.")
    exit()

all_global_data = []

# Wipe the summary file clean for a new run
with open(SUMMARY_FILE, 'w') as f:
    f.write("==================================================\n")
    f.write("      QUANTITATIVE FVG STRATEGY SCORECARD\n")
    f.write("==================================================\n\n")

print(f"Found {len(csv_files)} markets. Starting automated backtesting...\n")

# --- INDIVIDUAL MARKET PROCESSING ---
for file_path in csv_files:
    asset_name = os.path.basename(file_path).replace('.csv', '')
    print(f"Processing: {asset_name}...")
    
    try:
        # Load and Clean
        df = pd.read_csv(file_path)
        data = df['Message'].str.split(',', expand=True)
        if data.shape[1] > 11:
            data = data.iloc[:, :11]
            
        data.columns = columns
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        data = data.dropna()
        
        # Save for global model
        data['asset'] = asset_name
        all_global_data.append(data)
        
        y = data['outcome']
        X = data[features]
        
        total_trades = len(data)
        baseline_wr = (y.sum() / len(y)) * 100 if len(y) > 0 else 0
        
        # Split Data
        split_index = int(len(data) * 0.8)
        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
        
        # Write to summary file
        with open(SUMMARY_FILE, 'a') as f:
            f.write(f"--- {asset_name} ---\n")
            f.write(f"Total Trades: {total_trades} | Baseline Win Rate: {baseline_wr:.2f}%\n")
        
        if len(y_train) == 0 or y_train.value_counts().get(1, 0) == 0:
            with open(SUMMARY_FILE, 'a') as f:
                f.write("  -> ERROR: Not enough winning trades to train AI.\n\n")
            continue
            
        # Imbalance Weights
        ratio = float(y_train.value_counts().get(0, 0)) / y_train.value_counts().get(1, 1)
        sample_weights = np.where(y_train == 1, ratio, 1.0)
        
        # Train Model
        model = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.7, min_samples_leaf=15, random_state=42)
        model.fit(X_train, y_train, sample_weight=sample_weights)
        
        # Predict on Unseen Test Data
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        test_wr = (y_test.sum() / len(y_test)) * 100 if len(y_test) > 0 else 0
        
        with open(SUMMARY_FILE, 'a') as f:
            f.write(f"Test Set Baseline: {test_wr:.2f}% (Out of {len(y_test)} trades)\n")
            
            for thresh in [0.65, 0.70, 0.75]:
                custom_preds = (y_pred_proba >= thresh).astype(int)
                trades_taken = custom_preds.sum()
                if trades_taken > 0:
                    win_rate = precision_score(y_test, custom_preds, zero_division=0) * 100
                    f.write(f"  > AI > {thresh*100:.0f}% Conf | Trades: {trades_taken:<3} | Win Rate: {win_rate:.2f}%\n")
                else:
                    f.write(f"  > AI > {thresh*100:.0f}% Conf | Trades: 0\n")
            f.write("\n")
            
    except Exception as e:
        print(f"Error on {asset_name}: {e}")

# --- GLOBAL MASTER AI PROCESSING ---
print("\nCompiling Master AI Model...")
with open(SUMMARY_FILE, 'a') as f:
    f.write("==================================================\n")
    f.write("            MASTER AI MODEL (ALL ASSETS)          \n")
    f.write("==================================================\n")

if all_global_data:
    combined_df = pd.concat(all_global_data, ignore_index=True)
    
    y = combined_df['outcome']
    X = combined_df[features]
    
    total_trades = len(combined_df)
    baseline_wr = (y.sum() / len(y)) * 100 if len(y) > 0 else 0
    
    split_index = int(len(combined_df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    ratio = float(y_train.value_counts().get(0, 0)) / y_train.value_counts().get(1, 1)
    sample_weights = np.where(y_train == 1, ratio, 1.0)
    
    # Train robust global model
    model = GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=42)
    model.fit(X_train, y_train, sample_weight=sample_weights)
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    test_wr = (y_test.sum() / len(y_test)) * 100 if len(y_test) > 0 else 0
    
    with open(SUMMARY_FILE, 'a') as f:
        f.write(f"Total Global Trades: {total_trades}\n")
        f.write(f"Global Baseline Win Rate: {baseline_wr:.2f}%\n")
        f.write(f"Master Test Set Baseline: {test_wr:.2f}% (Out of {len(y_test)} possible setups)\n\n")
        
        for thresh in [0.60, 0.65, 0.70, 0.75, 0.80]:
            custom_preds = (y_pred_proba >= thresh).astype(int)
            trades_taken = custom_preds.sum()
            if trades_taken > 0:
                win_rate = precision_score(y_test, custom_preds, zero_division=0) * 100
                f.write(f"  > Master AI > {thresh*100:.0f}% Conf | Trades Approved: {trades_taken:<4} | Win Rate: {win_rate:.2f}%\n")
            else:
                f.write(f"  > Master AI > {thresh*100:.0f}% Conf | Trades Approved: 0\n")
                
        # Feature Importance
        importances = model.feature_importances_
        fi_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
        f.write("\n--- WHAT THE MASTER AI CARES ABOUT MOST ---\n")
        f.write(fi_df.to_string(index=False))

print(f"\nSuccess! Check the '{SUMMARY_FILE}' in your folder for the complete breakdown.")