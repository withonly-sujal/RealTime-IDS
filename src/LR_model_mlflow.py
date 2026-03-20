import joblib
import pandas as pd
import numpy as np
import mlflow
from pathlib import Path
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


# PATHS
BASE_DIR = Path(__file__).resolve().parent.parent

# Base models
XGB_PROCESSED_PATH = BASE_DIR / "models" / "IDS_XGBoost_Model_v3.pkl"
XGB_SELECTED_PATH = BASE_DIR / "models" / "IDS_XGBoost_Model_v6.pkl"

MLP_PROCESSED_PATH = BASE_DIR / "models" / "IDS_MLP_Model_v1.pkl"
MLP_SELECTED_PATH = BASE_DIR / "models" / "IDS_MLP_Model_v3.pkl"

# Datasets
TRAIN_PROCESSED = BASE_DIR / "data" / "processed" / "train_processed_balanced.csv"
TRAIN_SELECTED = BASE_DIR / "data" / "processed" / "train_selected_balanced.csv"

TEST_PROCESSED = BASE_DIR / "data" / "processed" / "test_processed.csv"
TEST_SELECTED = BASE_DIR / "data" / "processed" / "test_selected.csv"


# LOAD DATA
train_proc = pd.read_csv(TRAIN_PROCESSED)
train_sel = pd.read_csv(TRAIN_SELECTED)

test_proc = pd.read_csv(TEST_PROCESSED)
test_sel = pd.read_csv(TEST_SELECTED)

print("Label alignment:", (train_proc["label"] == train_sel["label"]).all())

X_train_proc = train_proc.drop(columns=["label"])
y_train = train_proc["label"]

X_train_sel = train_sel.drop(columns=["label"])

X_test_proc = test_proc.drop(columns=["label"])
y_test = test_proc["label"]

X_test_sel = test_sel.drop(columns=["label"])


# SPLIT (NO DATA LEAKAGE)
X_tr_proc, X_val_proc, y_tr, y_val = train_test_split(
    X_train_proc, y_train, test_size=0.2, random_state=42
)

X_tr_sel, X_val_sel, _, _ = train_test_split(
    X_train_sel, y_train, test_size=0.2, random_state=42
)



# LOAD MODELS

# XGB
xgb_proc = joblib.load(XGB_PROCESSED_PATH)
xgb_sel = joblib.load(XGB_SELECTED_PATH)

# MLP
mlp_proc_loaded = joblib.load(MLP_PROCESSED_PATH)
mlp_sel_loaded = joblib.load(MLP_SELECTED_PATH)

mlp_proc = mlp_proc_loaded["model"]
scaler_proc = mlp_proc_loaded["scaler"]

mlp_sel = mlp_sel_loaded["model"]
scaler_sel = mlp_sel_loaded["scaler"]



# VALIDATION PREDICTIONS
xgb_proc_val = xgb_proc.predict_proba(X_val_proc)[:, 1]
xgb_sel_val = xgb_sel.predict_proba(X_val_sel)[:, 1]

X_val_proc_scaled = scaler_proc.transform(X_val_proc)
X_val_sel_scaled = scaler_sel.transform(X_val_sel)

mlp_proc_val = mlp_proc.predict(X_val_proc_scaled).ravel()
mlp_sel_val = mlp_sel.predict(X_val_sel_scaled).ravel()


# STACK FEATURES
stack_val = np.column_stack([
    xgb_proc_val,
    xgb_sel_val,
    mlp_proc_val,
    mlp_sel_val
])


# META MODEL (ONLY LR)
meta_model = LogisticRegression()


# MLFLOW
mlflow.set_experiment("IDS_Stacking_Final")

with mlflow.start_run(run_name="Stack_LogisticRegression_FINAL"):

    print("\nTraining Logistic Regression meta model...")

    # Train
    meta_model.fit(stack_val, y_val)


    # TEST PREDICTIONS
    xgb_proc_test = xgb_proc.predict_proba(X_test_proc)[:, 1]
    xgb_sel_test = xgb_sel.predict_proba(X_test_sel)[:, 1]

    X_test_proc_scaled = scaler_proc.transform(X_test_proc)
    X_test_sel_scaled = scaler_sel.transform(X_test_sel)

    mlp_proc_test = mlp_proc.predict(X_test_proc_scaled).ravel()
    mlp_sel_test = mlp_sel.predict(X_test_sel_scaled).ravel()

    stack_test = np.column_stack([
        xgb_proc_test,
        xgb_sel_test,
        mlp_proc_test,
        mlp_sel_test
    ])

    y_prob = meta_model.predict_proba(stack_test)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)


    # METRICS
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()

    print("Accuracy:", accuracy)
    print("F1:", f1)


    # MLFLOW LOGGING
    mlflow.log_param("meta_model", "LogisticRegression")

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)

    mlflow.log_metric("TN", tn)
    mlflow.log_metric("FP", fp)
    mlflow.log_metric("FN", fn)
    mlflow.log_metric("TP", tp)

    mlflow.log_metric("ADR", tp / (tp + fn))
    mlflow.log_metric("FAR", fp / (fp + tn))


# SAVE FINAL MODEL
os.makedirs(BASE_DIR / "models", exist_ok=True)

final_model_path = BASE_DIR / "models" / "IDS_Stacking_LogisticRegression.pkl"

joblib.dump({
    "meta_model": meta_model,
    "xgb_proc": xgb_proc,
    "xgb_sel": xgb_sel,
    "mlp_proc": mlp_proc,
    "mlp_sel": mlp_sel,
    "scaler_proc": scaler_proc,
    "scaler_sel": scaler_sel
}, final_model_path)

print(f"\nFinal stacking model saved at: {final_model_path}")
print("\nStacking completed successfully")