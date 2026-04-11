import optuna

study = optuna.load_study(
    study_name="sequential_fine_tuning", 
    storage="sqlite:///sequential_fine_tuning.db"
)

running_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING]
for trial in running_trials:
    # Mark the stuck trial as failed
    study.storage.set_trial_state(trial._trial_id, optuna.trial.TrialState.FAIL)
    # Enqueue the exact same hyperparameters to be tried again
    study.enqueue_trial(trial.params)
