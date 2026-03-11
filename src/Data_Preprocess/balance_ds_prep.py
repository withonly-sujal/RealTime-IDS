import pandas as pd
from pathlib import Path
from sklearn.utils import resample


# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent

PROCESSED_TRAIN = BASE_DIR / "data" / "processed" / "train_processed.csv"
SELECTED_TRAIN = BASE_DIR / "data" / "processed" / "train_selected.csv"

PROCESSED_OUT = BASE_DIR / "data" / "processed" / "train_processed_balanced.csv"
SELECTED_OUT = BASE_DIR / "data" / "processed" / "train_selected_balanced.csv"


def balance_dataset(input_path, output_path):

    print(f"\nLoading dataset: {input_path}")

    df = pd.read_csv(input_path)

    # Split classes
    df_attack = df[df["label"] == 1]
    df_normal = df[df["label"] == 0]

    print("Original distribution:")
    print(df["label"].value_counts())

    # Determine minority class size
    min_size = min(len(df_attack), len(df_normal))

    # Downsample majority class
    df_attack_balanced = resample(
        df_attack,
        replace=False,
        n_samples=min_size,
        random_state=42
    )

    df_normal_balanced = resample(
        df_normal,
        replace=False,
        n_samples=min_size,
        random_state=42
    )

    # Combine
    df_balanced = pd.concat([df_attack_balanced, df_normal_balanced])

    # Shuffle
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    print("\nBalanced distribution:")
    print(df_balanced["label"].value_counts())

    # Save
    df_balanced.to_csv(output_path, index=False)

    print(f"Balanced dataset saved to: {output_path}")


# Run balancing
if __name__ == "__main__":

    balance_dataset(PROCESSED_TRAIN, PROCESSED_OUT)
    balance_dataset(SELECTED_TRAIN, SELECTED_OUT)