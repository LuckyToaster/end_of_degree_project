import optuna
import plotly.graph_objects as go
from pathlib import Path


def get_best_trial(study_name, storage_path):
    try: study = optuna.load_study(study_name=study_name, storage=storage_path)
    except: 
        print(f'Could not load study {study_name}')
        return

    try: return study.best_trial
    except: 
        print(f'No completed trials found for {study_name}')
        return


def plot_study_losses(study_name, storage_path, output_dir):
    """
    Retrieves the best trial from an Optuna study, extracts the train and validation 
    losses saved as user attributes, and generates a Plotly graph.
    """
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_path)
    except Exception as e:
        print(f"Could not load study {study_name}: {e}")
        return

    try:
        best_trial = study.best_trial
    except ValueError:
        print(f"No completed trials found for {study_name}")
        return

    train_losses = best_trial.user_attrs.get("train_losses", [])
    val_losses = best_trial.user_attrs.get("val_losses", [])

    if not train_losses or not val_losses:
        print(f"No loss data found for the best trial in {study_name}")
        return

    # Extract the average loss (the last element in each epoch's list)
    train_avg_losses = [epoch[-1] for epoch in train_losses]
    val_avg_losses = [epoch[-1] for epoch in val_losses]
    epochs = list(range(1, len(train_avg_losses) + 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs, y=train_avg_losses, mode='lines+markers', name='Train Loss',
        line=dict(color='black', dash='dash'), 
        marker=dict(color='black')
    ))
    fig.add_trace(go.Scatter(
        x=epochs, y=val_avg_losses, mode='lines+markers', name='Validation Loss',
        line=dict(color='black'), 
        marker=dict(color='white', line=dict(color='black', width=1.5))
    ))

    fig.update_layout(
        title=f'Loss vs Epochs for {study_name} (Best Trial: {best_trial.number})',
        xaxis_title='Epoch',
        yaxis_title='Average Loss',
        yaxis=dict(range=[1, 0]),
        template='plotly_white'
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / f"{study_name}_best_trial_loss.png"
    
    # Write image (requires kaleido to be installed)
    try:
        fig.write_image(str(output_path))
        print(f"Saved plot for {study_name} to {output_path}")
    except ValueError as e:
        print(f"Could not save image (is kaleido installed?): {e}")


def main():
    db_dir = "data/studies"
    output_dir = "data/plots"

    plot_study_losses(
        study_name='model_shootout', 
        storage_path=f'sqlite:///{db_dir}/model_shootout.db',
        output_dir=output_dir
    )
    
    # plot_study_losses(
    #     study_name='sequential_fine_tuning', 
    #     storage_path=f'sqlite:///{db_dir}/sequential_fine_tuning.db',
    #     output_dir=output_dir
    # )
    

