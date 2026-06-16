import pandas as pd
from src.constants import CSV_PATH
from src.helpers.ml import three_way_split

TARGETS = ['fat_g', 'carb_g', 'prot_g']
train_df, val_df, _ = three_way_split(CSV_PATH, TARGETS, 1)
print(f"Train length: {len(train_df)}")
print(f"Val length: {len(val_df)}")
