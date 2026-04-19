from pathlib import Path

# Root directory of project
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Paths
MODELS_DIR = BASE_DIR / "saved_models"

SELECTED_FEATURES_PATH = MODELS_DIR / "features_selected.txt"
PROCESSED_FEATURES_PATH = MODELS_DIR / "features_processed.txt"

STACK_MODEL_PATH = MODELS_DIR / "IDS_Stacking_LogisticRegression.pkl"