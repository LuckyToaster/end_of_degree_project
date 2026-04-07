import optuna
from tabulate import tabulate
import pandas as pd

def pprint(dataframe, showindex=False):
    print(tabulate(dataframe, headers='keys', showindex=showindex))

study = optuna.load_study(
    study_name="model_shootout_v1", 
    storage="sqlite:///model_shootout.db"
)

df = study.trials_dataframe()
print(df.sort_values("value").head()) # See best trials
