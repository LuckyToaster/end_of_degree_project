import pandas as pd
from src.constants import CSV_PATH
from src.helpers.ml import three_way_split

TARGETS = ['fat_g', 'carb_g', 'prot_g']
train_df, val_df, hidden_df = three_way_split(CSV_PATH, TARGETS, 1)
print("Train stats:")
print(train_df[TARGETS].mean())
print(train_df[TARGETS].std())
print("\nVal stats:")
print(val_df[TARGETS].mean())
print(val_df[TARGETS].std())

