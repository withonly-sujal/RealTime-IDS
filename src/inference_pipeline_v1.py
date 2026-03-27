import joblib
import pandas as pd
import numpy as np
import pathlib as path

# PATHS
BASE_DIR = path.Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "IDS_Stacking_LogisticRegression.pkl"
INPUT_PATH = BASE_DIR / "data" / "processed" / "test_processed.csv"

# LOAD MODEL
print("Loading model...")

loaded = joblib.load(MODEL_PATH)

meta_model = loaded["meta_model"]

xgb_proc = loaded["xgb_proc"]
xgb_sel = loaded["xgb_sel"]

mlp_proc = loaded["mlp_proc"]
mlp_sel = loaded["mlp_sel"]

scaler_proc = loaded["scaler_proc"]
scaler_sel = loaded["scaler_sel"]

# Feature lists (VERY IMPORTANT)
features_proc = loaded["features_proc"]
features_sel = loaded["features_sel"]

print("Model loaded successfully")


# LOAD INPUT DATA
print("Loading input data...")

df = pd.read_csv(INPUT_PATH)

# Remove label if exists
if "label" in df.columns:
    df = df.drop(columns=["label"])

print(f"Input shape: {df.shape}")


# FEATURE ALIGNMENT

# Check missing columns
missing_proc = set(features_proc) - set(df.columns)
missing_sel = set(features_sel) - set(df.columns)

if missing_proc:
    raise ValueError(f"Missing processed features: {missing_proc}")

if missing_sel:
    raise ValueError(f"Missing selected features: {missing_sel}")

# Ensure correct order
X_proc = df[features_proc]
X_sel = df[features_sel]


# BASE MODEL PREDICTIONS
print("Running base models...")

# XGBoost
xgb_proc_prob = xgb_proc.predict_proba(X_proc)[:, 1]
xgb_sel_prob = xgb_sel.predict_proba(X_sel)[:, 1]

# MLP (with scaling)
X_proc_scaled = scaler_proc.transform(X_proc)
X_sel_scaled = scaler_sel.transform(X_sel)

mlp_proc_prob = mlp_proc.predict(X_proc_scaled).ravel()
mlp_sel_prob = mlp_sel.predict(X_sel_scaled).ravel()


# STACKING INPUT
stack_input = np.column_stack([
    xgb_proc_prob,
    xgb_sel_prob,
    mlp_proc_prob,
    mlp_sel_prob
])


# FINAL PREDICTION
print("Running stacking model...")

y_prob = meta_model.predict_proba(stack_input)[:, 1]

# Threshold (can tune later)
threshold = 0.5

y_pred = (y_prob > threshold).astype(int)


# OUTPUT
results = pd.DataFrame({
    "Prediction": y_pred,
    "Probability": y_prob
})

print("\nSample Predictions:")
print(results.head())


# Save results
OUTPUT_PATH = BASE_DIR / "results" / "predictions.csv"
results.to_csv(OUTPUT_PATH, index=False)

print(f"\nResults saved to {OUTPUT_PATH}")
print("\nInference complete ")