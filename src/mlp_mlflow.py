import os
import joblib
import pandas as pd
import mlflow
import mlflow.keras
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import StandardScaler
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
def get_next_model_version(model_name="IDS_MLP_Model"):
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
TRAIN_PATH = BASE_DIR / "data" / "processed" / "train_selected_balanced.csv"  # change dataset here


# Load Dataset
print("Loading dataset...")

df = pd.read_csv(TRAIN_PATH)

X = df.drop(columns=["label"])
y = df["label"]


# Train-Val Split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ---------------------------
# Feature Scaling (important for neural networks)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)


# ---------------------------
# Build MLP Model

input_dim = X_train.shape[1]

model = Sequential([
    Dense(128, activation="relu", input_shape=(input_dim,)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation="relu"),
    BatchNormalization(),
    Dropout(0.2),

    Dense(32, activation="relu"),

    Dense(1, activation="sigmoid")
])


model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ---------------------------
# MLflow Experiment

mlflow.set_experiment("IDS_MLP_Model_Experiment")

with mlflow.start_run():

    print("Training MLP model...")

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=256,
        verbose=1
    )


    # Predictions
    y_prob = model.predict(X_val).ravel()
    y_pred = (y_prob > 0.5).astype(int)


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


    # ---------------------------
    # Log Parameters

    mlflow.log_param("model_type", "MLP")
    mlflow.log_param("layers", "128-64-32")
    mlflow.log_param("activation", "relu")
    mlflow.log_param("dropout_1", 0.3)
    mlflow.log_param("dropout_2", 0.2)
    mlflow.log_param("optimizer", "Adam")
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("batch_size", 256)
    mlflow.log_param("epochs", 30)

    mlflow.log_param("dataset", TRAIN_PATH.name)


    # ---------------------------
    # Log Metrics

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)


    tn, fp, fn, tp = cm.ravel()

    mlflow.log_metric("TN", tn)
    mlflow.log_metric("FP", fp)
    mlflow.log_metric("FN", fn)
    mlflow.log_metric("TP", tp)


    # ---------------------------
    # Confusion Matrix Plot

    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path)

    mlflow.log_artifact(cm_path)

    plt.close()


    # ---------------------------
    # ROC Curve

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


    # ---------------------------
    # Log model to MLflow

    mlflow.keras.log_model(
        model,
        artifact_path="model",
        registered_model_name="IDS_MLP_Model"
    )


    # ---------------------------
    # Save model locally

    version = get_next_model_version("IDS_MLP_Model")

    model_filename = f"IDS_MLP_Model_v{version}.pkl"
    model_path = os.path.join("models", model_filename)

    joblib.dump({
    "model": model,
    "scaler": scaler
    }, model_path)

    print(f"Model saved locally as: {model_filename}")


print("Training complete. Model logged to MLflow.")