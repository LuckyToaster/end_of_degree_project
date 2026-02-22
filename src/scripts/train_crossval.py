from sklearn.model_selection import KFold
import pandas as pd
from src.mmfood100k.dataset import MMFood100KDataset

df = pd.read_csv('data/mm-food-100k/mm-food-100k.csv')
ds = MMFood100KDataset(df)


