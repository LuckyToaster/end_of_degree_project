import numpy as np
import pandas as pd
from src.helpers.ml import three_way_split
from src.constants import CSV_PATH

TARGETS = ['fat_g', 'carb_g', 'prot_g']
train_df, val_df, hidden_df = three_way_split(CSV_PATH, TARGETS, 1)

print("Train MAE if predicting 0:", np.mean(np.abs(train_df[TARGETS].values)))
print("Val MAE if predicting 0:", np.mean(np.abs(val_df[TARGETS].values)))
