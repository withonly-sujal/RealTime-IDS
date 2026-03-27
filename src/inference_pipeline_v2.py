import joblib
import pandas as pd
import numpy as np
import pathlib as path


# PATHS
BASE_DIR = path.Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "IDS_Stacking_LogisticRegression.pkl"
INPUT_PATH = BASE_DIR / "data" / "processed" / "train_processed_balanced.csv"


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

features_proc = loaded["features_proc"]
features_sel = loaded["features_sel"]

print("Model loaded successfully")


# LOAD DATA (WITH LABEL)
print("Loading dataset...")

df = pd.read_csv(INPUT_PATH)

# Save true labels
y_true = df["label"]

# Drop label BEFORE inference
df = df.drop(columns=["label"])

print(f"Input shape: {df.shape}")


# FEATURE ALIGNMENT
missing_proc = set(features_proc) - set(df.columns)
missing_sel = set(features_sel) - set(df.columns)

if missing_proc:
    raise ValueError(f"Missing processed features: {missing_proc}")

if missing_sel:
    raise ValueError(f"Missing selected features: {missing_sel}")

X_proc = df[features_proc]
X_sel = df[features_sel]



# BASE MODEL PREDICTIONS
print("Running base models...")

xgb_proc_prob = xgb_proc.predict_proba(X_proc)[:, 1]
xgb_sel_prob = xgb_sel.predict_proba(X_sel)[:, 1]

X_proc_scaled = scaler_proc.transform(X_proc)
X_sel_scaled = scaler_sel.transform(X_sel)

mlp_proc_prob = mlp_proc.predict(X_proc_scaled).ravel()
mlp_sel_prob = mlp_sel.predict(X_sel_scaled).ravel()



# STACKING


stack_input = np.column_stack([
    xgb_proc_prob,
    xgb_sel_prob,
    mlp_proc_prob,
    mlp_sel_prob
])



# FINAL PREDICTION
print("Running stacking model...")

y_prob = meta_model.predict_proba(stack_input)[:, 1]

threshold = 0.4
y_pred = (y_prob > threshold).astype(int)



# SAVE RESULTS (WITH TRUE LABELS)
results = pd.DataFrame({
    "True_Label": y_true,
    "Prediction": y_pred,
    "Probability": y_prob
})

OUTPUT_PATH = BASE_DIR / "inference_pipeline_results" / "predictions_with_labels_balanced_DS_1.csv"
results.to_csv(OUTPUT_PATH, index=False)

print("\nSample Predictions:")
print(results.head())

print(f"\nSaved to {OUTPUT_PATH}")
print("\nInference complete")