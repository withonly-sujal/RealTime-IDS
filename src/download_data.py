import os
import gdown

DATA_DIR = "data/processed"

os.makedirs(DATA_DIR, exist_ok=True)

train_url = "https://drive.google.com/file/d/1SrgTcH4zm9UelEielOPDgoAYjQ4NblZ0/view?usp=drive_link"
test_url = "https://drive.google.com/file/d/1pmzmbJWW04G6tiQWISvg6bITuaSM4CEE/view?usp=drive_link"

train_path = f"{DATA_DIR}/train_selected.csv"
test_path = f"{DATA_DIR}/test_selected.csv"

if not os.path.exists(train_path):
    print("Downloading train dataset...")
    gdown.download(train_url, train_path, quiet=False)

if not os.path.exists(test_path):
    print("Downloading test dataset...")
    gdown.download(test_url, test_path, quiet=False)

print("Dataset ready.")