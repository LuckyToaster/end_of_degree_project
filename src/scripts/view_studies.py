import optuna
from tabulate import tabulate
# from sys import argv

def pprint(dataframe, showindex=False):
    print(tabulate(dataframe, headers='keys', showindex=showindex))

model_shootout_study = optuna.load_study(
    study_name="model_shootout_v1", 
    storage="sqlite:///model_shootout.db"
)

sequential_fine_tuning_study = optuna.load_study(
    study_name="sequential_fine_tuning", 
    storage="sqlite:///sequential_fine_tuning.db"
)

df1 = model_shootout_study.trials_dataframe()
print('model shootout')
print(df1.sort_values("value")) # See best trials

df2 = sequential_fine_tuning_study.trials_dataframe()
print('sequential fine tuning study')
print(df2.sort_values("value")) # See best trials

