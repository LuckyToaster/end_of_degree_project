import optuna
for s in optuna.get_all_study_summaries(storage='sqlite:///data/studies/model_shootout.db'):
    print(s.study_name)
