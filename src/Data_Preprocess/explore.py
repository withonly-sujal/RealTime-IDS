import pandas as pd

train = pd.read_csv("data/processed/train_processed_5dropped.csv")

print(train.shape)
print(train.columns)
print("train_processed_5dropped.csv")