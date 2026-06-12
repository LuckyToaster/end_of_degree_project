from pathlib import Path
import optuna, sys
import plotly.graph_objects as go
from src.constants import PLOTS_DIR, STUDIES_DIR


def plot_best_trial_losses(study_name, storage_path, dst_path):
    best_trial = optuna.load_study(study_name=study_name, storage=storage_path).best_trial
    train_losses = best_trial.user_attrs.get("train_losses", [])
    val_losses = best_trial.user_attrs.get("val_losses", [])

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
    fig.write_image(dst_path)


def main():
    if len(sys.argv) > 1:
        args = sys.argv[1].split('.')
        Path(PLOTS_DIR).mkdir(exist_ok=True, parents=True)
        plot_best_trial_losses(
            study_name=args[0], 
            storage_path=f'sqlite:///{STUDIES_DIR}/{args[1]}.db',
            dst_path=f'{PLOTS_DIR}/{args[0]}.png'
        )
