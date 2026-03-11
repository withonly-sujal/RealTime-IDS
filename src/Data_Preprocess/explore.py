import pandas as pd

train = pd.read_csv("data/raw/UNSW_NB15_training-set.csv")

print(train.shape)
print(train.columns)
print(train.info())
print(train['label'].value_counts())