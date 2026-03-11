import os
import joblib
import pandas as pd
import mlflow
import mlflow.xgboost
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_score,
    confusion_matrix,
    roc_curve
)
from sklearn.model_selection import train_test_split

# Helper function for model versioning
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


# Dataset Path
BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / "data" / "processed" / "train_processed.csv"

# Load Dataset
print("Loading dataset...")

df = pd.read_csv(TRAIN_PATH)

X = df.drop(columns=["label"])
y = df["label"]

# Train-Val Split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#---------------------------
# MLflow Experiment
# This is where it all starts with MLFlow. We set up an experiment and log everything here

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

    # Predictions
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    # Evaluation Metrics
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_prob)
    cm = confusion_matrix(y_val, y_pred)


    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)
    print("ROC-AUC:", roc_auc)
    print("Confusion Matrix:\n", cm)


    # Log Parameters
    mlflow.log_param("n_estimators", model.get_params()["n_estimators"])
    mlflow.log_param("max_depth", model.get_params()["max_depth"])
    mlflow.log_param("learning_rate", model.get_params()["learning_rate"])
    mlflow.log_param("subsample", model.get_params()["subsample"])
    mlflow.log_param("colsample_bytree", model.get_params()["colsample_bytree"])
    mlflow.log_param("eval_metric", model.get_params()["eval_metric"])
    mlflow.log_param("random_state", model.get_params()["random_state"])
    mlflow.log_param("n_jobs", model.get_params()["n_jobs"])
    
    # Log scale_pos_weight if it exists (for imbalanced datasets)
    mlflow.log_param("scale_pos_weight", model.get_params().get("scale_pos_weight", "None"))
    # Log Dataset Used
    mlflow.log_param("dataset", TRAIN_PATH.name)


    # Log Metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)


    # Log confusion matrix values
    tn, fp, fn, tp = cm.ravel()
    mlflow.log_metric("TN", tn)
    mlflow.log_metric("FP", fp)
    mlflow.log_metric("FN", fn)
    mlflow.log_metric("TP", tp)


    # -------- Confusion Matrix Plot --------
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path)
    plt.close()


    # -------- ROC Curve --------
    fpr, tpr, _ = roc_curve(y_val, y_prob)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    roc_path = "roc_curve.png"
    plt.savefig(roc_path)
    mlflow.log_artifact(roc_path)
    plt.close()

    # Log model to MLflow Registry
    mlflow.xgboost.log_model(
        model,
        artifact_path="model",
        registered_model_name="IDS_XGBoost_Model"
    )

    # Save model locally with version
    version = get_next_model_version("IDS_XGBoost_Model")

    model_filename = f"IDS_XGBoost_Model_v{version}.pkl"
    model_path = os.path.join("models", model_filename)

    joblib.dump(model, model_path)

    print(f"Model saved locally as: {model_filename}")

print("Training complete. Model logged to MLflow.")