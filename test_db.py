import optuna
study = optuna.load_study(study_name='shootout_EfficientNet_B3', storage='sqlite:///data/studies/model_shootout.db')
best = study.best_trial
print("Best trial:", best.number)
print("Val attr:", best.user_attrs.get("val_losses", [])[-1])
print("Train attr:", best.user_attrs.get("train_losses", [])[-1])
