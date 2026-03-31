import joblib
import pandas as pd
import numpy as np
import mlflow
from pathlib import Path
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


# ================= PATHS =================

BASE_DIR = Path(__file__).resolve().parent.parent


XGB_PROCESSED_PATH = BASE_DIR / "saved_models" / "IDS_XGBoost_Model_v8.pkl"
XGB_SELECTED_PATH = BASE_DIR / "saved_models" / "IDS_XGBoost_Model_v9.pkl"

MLP_PROCESSED_PATH = BASE_DIR / "saved_models" / "IDS_MLP_Model_v5.pkl"
MLP_SELECTED_PATH = BASE_DIR / "saved_models" / "IDS_MLP_Model_v7.pkl"

# DATASETS
TRAIN_PROCESSED_IMB = BASE_DIR / "data" / "processed" / "train_processed_5dropped.csv"
TRAIN_SELECTED_IMB = BASE_DIR / "data" / "processed" / "train_selected.csv"

TRAIN_PROCESSED_BAL = BASE_DIR / "data" / "processed" / "train_processed_balanced_5dropped.csv"
TRAIN_SELECTED_BAL = BASE_DIR / "data" / "processed" / "train_selected_balanced.csv"

TEST_PROCESSED = BASE_DIR / "data" / "processed" / "test_processed_5dropped.csv"
TEST_SELECTED = BASE_DIR / "data" / "processed" / "test_selected.csv"


# ================= LOAD =================

train_proc_imb = pd.read_csv(TRAIN_PROCESSED_IMB)
train_sel_imb = pd.read_csv(TRAIN_SELECTED_IMB)

train_proc_bal = pd.read_csv(TRAIN_PROCESSED_BAL)
train_sel_bal = pd.read_csv(TRAIN_SELECTED_BAL)

test_proc = pd.read_csv(TEST_PROCESSED)
test_sel = pd.read_csv(TEST_SELECTED)


# ================= SPLIT (ONLY ON IMBALANCED) =================

X_proc = train_proc_imb.drop(columns=["label"])
X_sel = train_sel_imb.drop(columns=["label"])
y = train_proc_imb["label"]

# SINGLE SPLIT
X_tr_proc, X_val_proc, y_tr, y_val = train_test_split(
    X_proc, y, test_size=0.2, random_state=42
)

# ALIGN selected using same indices
X_tr_sel = X_sel.iloc[X_tr_proc.index]
X_val_sel = X_sel.iloc[X_val_proc.index]


# ================= LOAD MODELS =================

xgb_proc = joblib.load(XGB_PROCESSED_PATH)
xgb_sel = joblib.load(XGB_SELECTED_PATH)

mlp_proc_loaded = joblib.load(MLP_PROCESSED_PATH)
mlp_sel_loaded = joblib.load(MLP_SELECTED_PATH)

mlp_proc = mlp_proc_loaded["model"]
scaler_proc = mlp_proc_loaded["scaler"]

mlp_sel = mlp_sel_loaded["model"]
scaler_sel = mlp_sel_loaded["scaler"]


# ================= VALIDATION =================

xgb_proc_val = xgb_proc.predict_proba(X_val_proc)[:, 1]
xgb_sel_val = xgb_sel.predict_proba(X_val_sel)[:, 1]

X_val_proc_scaled = scaler_proc.transform(X_val_proc)
X_val_sel_scaled = scaler_sel.transform(X_val_sel)

mlp_proc_val = mlp_proc.predict(X_val_proc_scaled).ravel()
mlp_sel_val = mlp_sel.predict(X_val_sel_scaled).ravel()


# ================= STACK =================

stack_val = np.column_stack([
    xgb_proc_val,
    xgb_sel_val,
    mlp_proc_val,
    mlp_sel_val
])


# ================= META MODEL =================

meta_model = LogisticRegression()

mlflow.set_experiment("IDS_Stacking_Final")

with mlflow.start_run(run_name="Stack_LogisticRegression_FINAL_v3"):

    meta_model.fit(stack_val, y_val)


    # ================= TEST =================

    X_test_proc = test_proc.drop(columns=["label"])
    X_test_sel = test_sel.drop(columns=["label"])
    y_test = test_proc["label"]

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
    y_pred = (y_prob > 0.4).astype(int)


    # ================= METRICS =================

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print("\n=== FINAL METRICS ===")
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)
    print("ROC AUC  :", roc_auc)
    print("Confusion Matrix:", cm)
    
    
    # ================= MLFLOW LOGGING =================
    
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
    
    # IDS-specific metrics
    mlflow.log_metric("ADR", tp / (tp + fn))
    mlflow.log_metric("FAR", fp / (fp + tn))

    
# ================= SAVE =================

final_model_path = BASE_DIR / "saved_models" / "IDS_Stacking_LogisticRegression_v3.pkl"

features_proc = train_proc_bal.drop(columns=["label"]).columns.tolist()
features_sel = train_sel_bal.drop(columns=["label"]).columns.tolist()

joblib.dump({
    "meta_model": meta_model,
    "xgb_proc": xgb_proc,
    "xgb_sel": xgb_sel,
    "mlp_proc": mlp_proc,
    "mlp_sel": mlp_sel,
    "scaler_proc": scaler_proc,
    "scaler_sel": scaler_sel,
    "features_proc": features_proc,
    "features_sel": features_sel
}, final_model_path)

print("\nStacking completed successfully")