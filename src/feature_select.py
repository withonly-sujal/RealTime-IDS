import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier


BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / "data" / "processed" / "train_processed.csv"
TEST_PATH = BASE_DIR / "data" / "processed" / "test_processed.csv"

SELECTED_DIR = BASE_DIR / "data" / "processed"
TRAIN_OUT = SELECTED_DIR / "train_selected.csv"
TEST_OUT = SELECTED_DIR / "test_selected.csv"

# loading the data
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

X_train = train.drop(columns=['label'])
y_train = train['label']

model = RandomForestClassifier(
    n_estimators = 100,
    random_state = 42,
    n_jobs = -1
)

model.fit(X_train, y_train)

importances = model.feature_importances_

feature_importance_df = pd.DataFrame({
    "feature": X_train.columns,
    "importance": importances
}).sort_values(by="importance", ascending = False)

print("\nTop 20 features:\n")
print(feature_importance_df.head(20))

TOP_N = 20
selected_features = feature_importance_df.head(TOP_N)["feature"].tolist()
print("\nSelected features:\n")
print(selected_features)

train_selected = train[selected_features + ['label']]
test_selected = test[selected_features + ['label']]

train_selected.to_csv(TRAIN_OUT, index=False)
test_selected.to_csv(TEST_OUT, index=False)

print("\nFeature selection complete.")
print("Saved:")
print(TRAIN_OUT)
print(TEST_OUT)