import pandas as pd
import joblib
from pathlib import Path

# PATHS
BASE_DIR = Path(__file__).resolve().parent.parent.parent

TRAIN_PROCESSED = BASE_DIR / "data" / "processed" / "train_processed_balanced.csv"
TRAIN_SELECTED = BASE_DIR / "data" / "processed" / "train_selected_balanced.csv"

SAVE_DIR = BASE_DIR / "models"
SAVE_DIR.mkdir(exist_ok=True)


# LOAD DATA
print("Loading datasets...")

df_proc = pd.read_csv(TRAIN_PROCESSED)
df_sel = pd.read_csv(TRAIN_SELECTED)


# EXTRACT FEATURES
features_proc = df_proc.drop(columns=["label"]).columns.tolist()
features_sel = df_sel.drop(columns=["label"]).columns.tolist()


# PRINT INFO
print("\nProcessed Features:", len(features_proc))
print("Selected Features:", len(features_sel))


# OPTIONAL CHECK
# Selected features should be subset of processed
missing = set(features_sel) - set(features_proc)

if len(missing) == 0:
    print("\n✅ Selected features are subset of processed features")
else:
    print("\n❌ WARNING: mismatch found:", missing)


# SAVE FEATURES (IMPORTANT)
joblib.dump(features_proc, SAVE_DIR / "features_processed.pkl")
joblib.dump(features_sel, SAVE_DIR / "features_selected.pkl")

print("\nFeature lists saved as .pkl")


# OPTIONAL: SAVE AS TXT
with open(SAVE_DIR / "features_processed.txt", "w") as f:
    for col in features_proc:
        f.write(col + "\n")

with open(SAVE_DIR / "features_selected.txt", "w") as f:
    for col in features_sel:
        f.write(col + "\n")

print("Feature lists also saved as .txt")

print("\nFeature extraction complete 🚀")