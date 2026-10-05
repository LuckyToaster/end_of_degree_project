import optuna, sys
from src.constants import STUDIES_DIR

def main():
    if len(sys.argv) > 1:
        args = sys.argv[1].split('.')
        study = optuna.load_study(study_name=args[1], storage=f'sqlite:///{STUDIES_DIR}/{args[0]}.db')
        print(study.trials_dataframe().sort_values('value'))
