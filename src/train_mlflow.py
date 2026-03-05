# src/train_mlflow.py
import os
import joblib
import pandas as pd
import mlflow
import mlflow.xgboost
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


# Helper function for versioning
def get_next_model_version(model_name="IDS_XGBoost_Model"):
    os.makedirs("models", exist_ok=True)

    versions = []

    for file in os.listdir("models"):
        if file.startswith(model_name) and file.endswith(".pkl"):
            try:
                v = int(file.split("_v")[-1].replace(".pkl", ""))
                versions.append(v)
            except:
                pass

    if len(versions) == 0:
        return 1

    return max(versions) + 1


# Paths to Dataset
BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / "data" / "processed" / "train_selected.csv"


# Loading Dataset
print("Loading dataset...")

df = pd.read_csv(TRAIN_PATH)

X = df.drop(columns=["label"])
y = df["label"]


# Train-Val Split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Starting MLflow Experiment
mlflow.set_experiment("IDS_XGBoost_Experiment")

with mlflow.start_run():

    print("Training XGBoost model...")

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    # Eval Metrics
    accuracy = accuracy_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_prob)

    print("Accuracy:", accuracy)
    print("Recall:", recall)
    print("F1:", f1)
    print("ROC-AUC:", roc_auc)

    # Log Parameters
    mlflow.log_param("model_type", "XGBoost")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 6)
    mlflow.log_param("learning_rate", 0.1)

    # Log Metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)

    # Model Logging to MLflow
    mlflow.xgboost.log_model(
        model,
        artifact_path="model",
        registered_model_name="IDS_XGBoost_Model"
    )

    # Save Model Locally with Version
    version = get_next_model_version("IDS_XGBoost_Model")

    model_filename = f"IDS_XGBoost_Model_v{version}.pkl"
    model_path = os.path.join("models", model_filename)

    joblib.dump(model, model_path)

    print(f"Model saved locally as: {model_filename}")


print("Training complete. Model logged to MLflow.")