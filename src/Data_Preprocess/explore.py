import pandas as pd

train = pd.read_csv("data/processed/train_selected.csv")

print(train.shape)
print(train.columns)
print("train_selected.csv")