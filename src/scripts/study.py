from argparse import ArgumentTypeError, ArgumentParser, Action
import optuna, sys, pydoc
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from src.constants import OPTUNA_DB_PATH, FIGURES_DIR

def trial_type(value):
    if value.lower() == 'best': return 'best'
    try: return int(value)
    except ValueError: raise ArgumentTypeError(f'Trial must be an integer or \'best\', got \'{value}\'')


def main():
    parser = ArgumentParser(description='View study trials and plot losses')
    parser.add_argument('study_name', help='Name of the Optuna study to view all trials')
    parser.add_argument('-t', '--trial', type=trial_type, default=None, help="Trial number")
    parser.add_argument("-p", "--plot-losses", action="store_true", help="Generate a plot in the figures directory")

    args = parser.parse_args()

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    study = optuna.load_study(study_name=args.study_name, storage=OPTUNA_DB_PATH)

    # if args.trial is None:
    #     pydoc.pager(study.trials_dataframe().sort_values('value'))
    #     sys.exit(0)
    #
    # if args.trial == 'best':
    #     trial = study.best_trial
    #     print(f'Number: {trial.number}\nValue: {trial.value}\nParams: {trial.params}')
    #     sys.exit(0)
    #
    # if args.trial >= len(study.trials): 
    #     raise ValueError(f'There are {len(study.trials)} trials in {args.study_name}. The largest --trial value permitted is {len(study.trials) - 1}')
    #
    # if args.trial:
    #     trial = study.trials[args.trial]
    #     print(f'Value: {trial.value}\nParams: {trial.params}')
    #     sys.exit(0)

    # guard
    if args.trial is None:
        pydoc.pager(study.trials_dataframe().sort_values('value'))
        sys.exit(0)

    # guard
    if type(args.trial) == int and args.trial >= len(study.trials):
        raise ValueError(f'There are {len(study.trials)} trials in {args.study_name}. The largest --trial value permitted is {len(study.trials) - 1}')

    # resolve
    trial = study.best_trial if args.trial == 'best' else study.trials[args.trial]

    # execute
    print(f'Number: {trial.number}\nValue: {trial.value}\nParams: {trial.params}')

    if args.plot_losses:
        Path(FIGURES_DIR).mkdir(exist_ok=True, parents=True)
            
        train_losses = trial.user_attrs.get("train_losses", [])
        val_losses = trial.user_attrs.get("val_losses", [])
        
        # if not train_losses or not val_losses:
        #     print(f"Warning: Trial {trial.number} does not have 'train_losses' or 'val_losses' user attributes.", file=sys.stderr)
        #     sys.exit(1)
            
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
            title=f'Loss vs Epochs for {args.study_name} (Trial: {trial.number})',
            xaxis_title='Epoch',
            yaxis_title='Average Loss',
            yaxis=dict(range=[0, 1]),
            template='plotly_white'
        )

        dst_path = f'{FIGURES_DIR}/{args.study_name}_trial_{trial.number}.png'
        fig.write_image(dst_path)
        print(f"Plot saved to {dst_path}")

