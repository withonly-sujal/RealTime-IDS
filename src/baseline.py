import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / "data" / "processed" / "train_processed.csv"


print("Loading dataset")
df = pd.read_csv(TRAIN_PATH)

print("Dataset Shape:", df.shape)

X = df.drop(columns=['label'])
y = df['label']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Baseline Random Forest model")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs = -1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_val)

print("\nAccuracy:", accuracy_score(y_val, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_val, y_pred))
