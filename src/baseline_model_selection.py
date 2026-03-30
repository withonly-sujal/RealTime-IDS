import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / "data" / "processed" / "train_selected.csv"


print("Loading dataset")
df = pd.read_csv(TRAIN_PATH)

print("Dataset Shape:", df.shape)
print(df['label'].value_counts())
X = df.drop(columns=['label'])
y = df['label']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Baseline Random Forest model")


# Random Forest Classifier with default parameters
#model = RandomForestClassifier(
#    n_estimators=100,
#    random_state=42,
#    n_jobs = -1
#)
#
#model.fit(X_train, y_train)
#
#y_pred = model.predict(X_val)
#y_proba = model.predict_proba(X_val)[:, 1]
#
#print("\nAccuracy:", accuracy_score(y_val, y_pred))
#
#print("\nRecall (Attack class=1):",
#      recall_score(y_val, y_pred, pos_label=1))
#
#print("F1 Score:",
#      f1_score(y_val, y_pred))
#
#print("ROC-AUC:",
#      roc_auc_score(y_val, y_proba))
#
#print("\nConfusion Matrix:")
#print(confusion_matrix(y_val, y_pred))
#
#print("\nClassification Report:\n")
#print(classification_report(y_val, y_pred))


# XGBoost Classifier with tuned parameters

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

# ----------------------------
# Predictions
# ----------------------------
y_pred = model.predict(X_val)
y_proba = model.predict_proba(X_val)[:, 1]

# ----------------------------
# Metrics
# ----------------------------
print("\nAccuracy:", accuracy_score(y_val, y_pred))

print("Recall (Attack class=1):",
      recall_score(y_val, y_pred, pos_label=1))

print("F1 Score:",
      f1_score(y_val, y_pred))

print("ROC-AUC:",
      roc_auc_score(y_val, y_proba))

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_val, y_pred))