import optuna
study = optuna.load_study(study_name='model_shootout', storage='sqlite:///data/studies/model_shootout.db')
best = study.best_trial
print("Train total losses per epoch:")
train_losses = best.user_attrs.get("train_losses", [])
for e in train_losses:
    print(e[-1])
print("\nVal total losses per epoch:")
val_losses = best.user_attrs.get("val_losses", [])
for e in val_losses:
    print(e[-1])
