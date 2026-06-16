import optuna
study = optuna.load_study(study_name='shootout_EfficientNet_B3', storage='sqlite:///data/studies/model_shootout.db')
print("Trial 0 state:", study.trials[0].state)
print("Trial 0 values:", study.trials[0].values)
if study.trials[0].state == optuna.trial.TrialState.FAIL:
    print("Trial 0 traceback:")
    print(study.trials[0].user_attrs.get("traceback", "No traceback found"))
