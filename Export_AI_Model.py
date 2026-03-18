import pandas as pd
import numpy as np
import os
import glob
import joblib
from sklearn.ensemble import GradientBoostingClassifier

# 1. MAKE SURE THIS POINTS TO YOUR 15min 1:3 RR FOLDER!
DATA_FOLDER = 'market_data_15min' 
features = ['dir', 'size_atr', 'overlap_pct', 'displacement', 'wick_ratio', 'dist_ema', 'pos_ema', 'session']

print("Loading data to train the Final Master AI...")
all_global_data = []
for file_path in glob.glob(os.path.join(DATA_FOLDER, '*.csv')):
    df = pd.read_csv(file_path)
    data = df['Message'].str.split(',', expand=True)
    if data.shape[1] > 11:
        data = data.iloc[:, :11]
    data.columns = ['dir', 'size_atr', 'overlap_pct', 'displacement', 'wick_ratio', 'dist_ema', 'pos_ema', 'session', 'outcome', 'bars_to_entry', 'bars_to_outcome']
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    all_global_data.append(data.dropna())

combined_df = pd.concat(all_global_data, ignore_index=True)
X = combined_df[features]
y = combined_df['outcome']

ratio = float(y.value_counts().get(0, 0)) / y.value_counts().get(1, 1)
sample_weights = np.where(y == 1, ratio, 1.0)

print("Training the 15min 1:3 RR Master Brain. This might take a few seconds...")

# 2. THIS IS THE UPDATED, ANTI-OVERFIT LINE
model = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.7, min_samples_leaf=15, random_state=42)

model.fit(X, y, sample_weight=sample_weights)

# Save the trained model to a file
joblib.dump(model, 'Master_AI_Brain.pkl')
print("Success! 'Master_AI_Brain.pkl' has been saved to your folder.")