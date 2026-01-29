
import fireducks.pandas as pd
from ml_helpers.misc import pprint

# https://huggingface.co/datasets/Codatta/MM-Food-100K
df = pd.read_csv("hf://datasets/Codatta/MM-Food-100K/MM-Food-100K.csv")

pprint(df.sample(n=10))
