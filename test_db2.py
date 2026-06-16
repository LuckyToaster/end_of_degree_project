import optuna
study = optuna.load_study(study_name='shootout_EfficientNet_B3', storage='sqlite:///data/studies/model_shootout.db')
for trial in study.trials:
    print(f"Trial {trial.number}: State {trial.state}, Value {trial.value}")
    if trial.state == optuna.trial.TrialState.COMPLETE:
        print("  Train:", trial.user_attrs.get("train_losses", [])[-1])
        print("  Val:", trial.user_attrs.get("val_losses", [])[-1])
