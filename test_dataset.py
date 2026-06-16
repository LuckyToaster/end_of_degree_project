import pandas as pd
from src.helpers.ml import three_way_split
from src.dataset import FoodDataset
from src.constants import CSV_PATH

TARGETS = ['fat_g', 'carb_g', 'prot_g']
train_df, val_df, hidden_df = three_way_split(CSV_PATH, TARGETS, 1)

ds = FoodDataset(train_df, transform=None, targets=TARGETS)
print(ds[0][1])
