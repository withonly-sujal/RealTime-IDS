# This script is dropping 5 specific features from the processed training dataset to create a new version with fewer features.

import pandas as pd
import pathlib as Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_PATH = BASE_DIR / "data" / "processed" / "train_processed.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "train_processed_5dropped.csv"

# ==== CONFIG ====
input_path = INPUT_PATH
output_path = OUTPUT_PATH

# Columns to drop
columns_to_drop = [
    "service",
    "trans_depth",
    "response_body_len",
    "is_ftp_login",
    "is_sm_ips_ports"
]

# ==== LOAD DATA ====
df = pd.read_csv(input_path)

print("Original shape:", df.shape)

# ==== DROP COLUMNS (only if they exist) ====
df_cleaned = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

print("New shape:", df_cleaned.shape)
print("Dropped columns:", [col for col in columns_to_drop if col in df.columns])

# ==== SAVE NEW DATASET ====
df_cleaned.to_csv(output_path, index=False)

print(f"✅ Cleaned dataset saved at: {output_path}")