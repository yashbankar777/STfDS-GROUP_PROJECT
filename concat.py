import pandas as pd
import glob
import os

DATA_DIR = "data/demand"
PATTERN = os.path.join(DATA_DIR, "PRICE_AND_DEMAND_*_NSW1.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "nem_price_demand_2022_2025.csv")

files = sorted(glob.glob(PATTERN))
print(f"Found {len(files)} CSV files to merge")

all_data = []
for f in files:
    df = pd.read_csv(f, skiprows=1)  # AEMO 'C' header
    all_data.append(df)
    print(f"  Loaded {os.path.basename(f)}: {len(df):,} rows")

full_df = pd.concat(all_data, ignore_index=True)
full_df['SETTLEMENTDATE'] = pd.to_datetime(full_df['SETTLEMENTDATE'])
full_df = full_df.sort_values('SETTLEMENTDATE')

full_df.to_csv(OUTPUT_FILE, index=False)
print(f"\nMerged dataset: {len(full_df):,} total rows")
print(f"Range: {full_df['SETTLEMENTDATE'].min()} → {full_df['SETTLEMENTDATE'].max()}")
print(f"Saved: {os.path.abspath(OUTPUT_FILE)}")