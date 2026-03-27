import pathlib as path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

# LOAD RESULTS

BASE_DIR = path.Path(__file__).resolve().parent.parent
RESULTS_PATH = BASE_DIR / "inference_pipeline_results" / "predictions_with_labels_balanced_DS_1.csv"
df = pd.read_csv(RESULTS_PATH)

y_true = df["True_Label"]
y_pred = df["Prediction"]
y_prob = df["Probability"]


# METRICS
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_prob)

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()



# PRINT RESULTS
print("\n===== FINAL EVALUATION =====")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print(f"\nTN: {tn}")
print(f"FP: {fp}")
print(f"FN: {fn}")
print(f"TP: {tp}")


# IDS METRICS
ADR = tp / (tp + fn)
FAR = fp / (fp + tn)

print("\n===== IDS METRICS =====")
print(f"Attack Detection Rate (ADR): {ADR:.4f}")
print(f"False Alarm Rate (FAR): {FAR:.4f}")