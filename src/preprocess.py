# Imports
import pandas as pd
import pathlib as Path
from sklearn.preprocessing import LabelEncoder

# Paths
BASE_DIR = Path.Path(__file__).parent.parent
TRAIN_PATH = BASE_DIR / "data" / "raw" / "UNSW_NB15_training-set.csv"
TEST_PATH = BASE_DIR / "data" / "raw" / "UNSW_NB15_testing-set.csv"

PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_OUT = PROCESSED_DIR / "train_processed.csv"
TEST_OUT = PROCESSED_DIR / "test_processed.csv"

# Loading the Data
print("Loading datasets...")
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

# Shape of the datasets
print("Train shape:", train.shape)
print("Test shape:", test.shape)

# Droping the Columns
#  drop_cols = ['id', 'attack_cat']
train.drop(columns=['id', 'attack_cat'], inplace=True)
test.drop(columns=['id', 'attack_cat'], inplace=True)

y_train = train['label']
y_test = test['label']
X_train = train.drop(columns=['label'])
X_test = test.drop(columns=['label'])

# Encode categorical data
cat_cols = ['proto', 'service', 'state']
print("Encoding categorical data")
encoders = {}

for col in cat_cols:
    le =LabelEncoder()
    combined_data = pd.concat([X_train[col], X_test[col]], axis=0)
    
    le.fit(combined_data)
    
    X_train[col] = le.transform(X_train[col])
    X_test[col] = le.transform(X_test[col])
    
    encoders[col] = le
    
# Combine Labels
train_processed = X_train.copy()
train_processed['label'] = y_train

test_processed = X_test.copy()
test_processed['label'] = y_test


# Save the processed datasets
train_processed.to_csv(TRAIN_OUT, index=False)
test_processed.to_csv(TEST_OUT, index=False)

print("Preprocessing complete.")
print("Processed files saved to:", PROCESSED_DIR)