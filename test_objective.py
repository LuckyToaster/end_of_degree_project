import optuna
from src.studies.model_shootout import objective

# Try running 1 epoch of the objective function to see if it crashes
study = optuna.create_study(direction="minimize")
try:
    objective(study.ask(), "EfficientNet_B3")
except Exception as e:
    import traceback
    traceback.print_exc()
