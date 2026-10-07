import optuna
import sys
from src.constants import OPTUNA_DB_PATH

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.scripts.view_study_trial <db_name>.<study_name>")
        print("Example: python -m src.scripts.view_best_trial model_shootout.shootout_Swin_V2_S")
        sys.exit(1)

    # args = sys.argv[1].split('.')
    # if len(args) != 2:
    #     print("Error: Argument must be in format <study_name>.<db_name>")
    #     sys.exit(1)
    # storage_url = f'sqlite:///{OPTUNA_DB_PATH}.db'
    
    try:
        study = optuna.load_study(study_name=sys.argv[1], storage=f'sqlite:///{OPTUNA_DB_PATH}.db')
    except Exception as e:
        print(f"Failed to load study: {e}")
        sys.exit(1)
        
    try:
        best_trial = study.best_trial
    except ValueError:
        print(f"Study '{sys.argv[1]}' has no completed trials to display.")
        sys.exit(1)

    print("=" * 40)
    print(f"Best Trial Information")
    print(f"Study: {sys.argv[1]}")
    print("=" * 40)
    print(f"Trial Number: {best_trial.number}")
    print(f"Value (Loss): {best_trial.value:.6f}")
    print("\nParameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
        
    if best_trial.user_attrs:
        print("\nUser Attributes (Summary):")
        for key, value in best_trial.user_attrs.items():
            if isinstance(value, list):
                print(f"  {key}: List of {len(value)} items")
            else:
                print(f"  {key}: {value}")

