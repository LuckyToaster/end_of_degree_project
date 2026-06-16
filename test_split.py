import pandas as pd
from src.helpers.ml import three_way_split
from src.constants import CSV_PATH

TARGETS = ['fat_g', 'carb_g', 'prot_g']
train, val, test = three_way_split(CSV_PATH, TARGETS, 1)
print(train[TARGETS].mean())
print(train[TARGETS].std())
