# src/test_model.py
import argparse
import joblib
import pandas as pd
import mlflow
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# Parse command line arguments
parser = argparse.ArgumentParser(description="Test IDS Model")
parser.add_argument(
    "--model",
    type=str,
    required=True,
    help="Path to trained .pkl model"
)

args = parser.parse_args()

MODEL_PATH = args.model

# Dataset Path
BASE_DIR = Path(__file__).resolve().parent.parent
TEST_PATH = BASE_DIR / "data" / "processed" / "test_selected.csv"

# Load model
print("Loading model...")
model = joblib.load(MODEL_PATH)
print("Model loaded:", MODEL_PATH)

# Load test dataset
print("Loading test dataset...")

df = pd.read_csv(TEST_PATH)

X_test = df.drop(columns=["label"])
y_test = df["label"]


# Predictions
print("Running predictions...")
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)


print("\n===== TEST RESULTS =====")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)
print("ROC-AUC  :", roc_auc)
print("\nConfusion Matrix:")
print(cm)



# MLflow Logging
mlflow.set_experiment("IDS_Model_Testing")
with mlflow.start_run():
    mlflow.log_param("model_path", MODEL_PATH)
    mlflow.log_param("model_type", "XGBoost")
    mlflow.log_param("dataset", "test_selected.csv")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall) 
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)
print("\nTest results logged to MLflow.")