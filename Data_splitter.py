import pandas as pd
import glob
import os

# Define folders
INPUT_FOLDER = 'market_data'
TRAIN_FOLDER = 'market_data_TRAIN'
TEST_FOLDER = 'market_data_TEST'

# Create the new folders if they don't exist
os.makedirs(TRAIN_FOLDER, exist_ok=True)
os.makedirs(TEST_FOLDER, exist_ok=True)

print("Splitting CSVs into Training (Old) and Backtesting (New) sets...")

for file_path in glob.glob(os.path.join(INPUT_FOLDER, '*.csv')):
    filename = os.path.basename(file_path)
    
    # Load the CSV
    df = pd.read_csv(file_path)
    
    # Calculate where the 80% mark is
    split_index = int(len(df) * 0.8)
    
    # Split the data
    train_df = df.iloc[:split_index] # Top 80% (Older data)
    test_df = df.iloc[split_index:]  # Bottom 20% (Recent Feb/March data)
    
    # Save to the new folders
    train_df.to_csv(os.path.join(TRAIN_FOLDER, filename), index=False)
    test_df.to_csv(os.path.join(TEST_FOLDER, filename), index=False)
    
    print(f"[{filename}] Split: {len(train_df)} Training rows | {len(test_df)} Backtest rows")

print("\nSuccess! Your data is now properly separated.")