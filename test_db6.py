import optuna
study = optuna.load_study(study_name='model_shootout', storage='sqlite:///data/studies/model_shootout.db')
best = study.best_trial
print("Best trial params:", best.params)
