from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from optuna import TrialPruned
import torch
import pandas as pd
from pandas import DataFrame
from tqdm.auto import tqdm

from typing import Optional
from os import makedirs

import numpy as np

from src.constants import MODEL_CHECKPOINTS_DIR

__all__ = [
    'dataloader_args',
    'standardize',
    'destandardize',
    'get_predictions'
    'train_eval_loop',
]

dataloader_args = dict(num_workers=4, pin_memory=True, persistent_workers=True)


def standardize(train_df, val_df, test_df, targets: list[str]):
    scaler = StandardScaler()
    scaler.set_output(transform="pandas")

    train_df[targets] = scaler.fit_transform(train_df[targets])
    val_df[targets] = scaler.transform(val_df[targets])
    test_df[targets] = scaler.transform(test_df[targets])

    means = scaler.mean_.tolist()
    stds = np.sqrt(scaler.var_).tolist()

    return train_df, val_df, test_df, means, stds


def destandardize(preds: torch.Tensor, means: list[float], stds: list[float]) -> torch.Tensor:
    means_t = torch.tensor(means, device=preds.device, dtype=preds.dtype)
    stds_t = torch.tensor(stds, device=preds.device, dtype=preds.dtype)
    return (preds * stds_t) + means_t


def three_way_split(
        csv_path: str, 
        seed: int = 1,
        split: tuple[float, float, float] = (0.9, 0.05, 0.05),
        stratify_target: Optional[str] = None,
) -> tuple[DataFrame, DataFrame, DataFrame]:

    if sum(split) != 1.0: 
        raise ValueError('split tuple items must add up to 1')

    df = pd.read_csv(csv_path)

    if stratify_target and stratify_target not in df.columns.tolist():
        raise ValueError("'stratify_target' is not a column in the df")

    if stratify_target:
        bins = pd.qcut(df[stratify_target], q=10, labels=False, duplicates='drop') # 1. Bin the target for stratification
        train_df, temp_df, _, temp_bins = train_test_split(df, bins, train_size=split[0], random_state=seed, stratify=bins)
        val_df, test_df = train_test_split(temp_df, train_size=split[1] / (split[1] + split[2]), random_state=seed, stratify=temp_bins)
    else: 
        train_df, temp_df = train_test_split(df, train_size=split[0], random_state=seed)
        val_df, test_df = train_test_split(temp_df, train_size=split[1] / (split[1] + split[2]), random_state=seed)

    return train_df, val_df, test_df


# def get_scaler_on_train(csv_path: str, targets: list[str], seed: int) -> StandardScaler:
#     df = pd.read_csv(csv_path)
#     train_df, _ = train_test_split(df, test_size=0.2, random_state=seed)
#     scaler = StandardScaler()
#     scaler.set_output(transform="pandas")
#     scaler.fit(train_df[targets])
#     return scaler


def get_predictions(loader, model):
    device = next(model.parameters()).device
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X, y in loader:
            all_preds.append(model(X.to(device)).cpu())
            all_targets.append(y)
    return torch.cat(all_preds), torch.cat(all_targets) # concatenate list of shape (batch_size, n_targets) into a single (batch_size*n_batches, n_targets) tensor


# get rid of the lasst redundant arguments
def train_eval_loop(model, train_loader, val_loader, epochs, criterion, optimizer, scheduler=None, trial=None, starting_epoch=0):
    device = next(model.parameters()).device # infer the device instead of passing it around all the time
    scaler = torch.amp.GradScaler('cuda')
    losses = {'train': [], 'val': []}

    for epoch in range(epochs): 
        actual_epoch = starting_epoch + epoch

        train_losses = train_epoch(train_loader, model, criterion, optimizer, scaler, actual_epoch)
        if scheduler is not None: scheduler.step()
        val_losses = validate_epoch(val_loader, model, criterion)

        if trial is not None:
            trial.report(val_losses['optuna_loss'], actual_epoch)
            trial.set_user_attr("train_losses", trial.user_attrs.get("train_losses", []) + [train_losses])
            trial.set_user_attr("val_losses", trial.user_attrs.get("val_losses", []) + [val_losses])
            if trial.should_prune(): raise TrialPruned()
        else:
            makedirs(MODEL_CHECKPOINTS_DIR, exist_ok=True)
            checkpoint_path = f'{MODEL_CHECKPOINTS_DIR}/{model.__class__.__name__}_epoch_{actual_epoch+1}.pt'
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Model checkpoint saved to {checkpoint_path}')

        losses['train'].append(train_losses)
        losses['val'].append(val_losses)
        print(f'Epoch {actual_epoch+1}: (Train {round(train_losses['backprop_loss'])}, raw {train_losses['optuna_loss']}), (Val {round(val_losses['backprop_loss'])}, raw {val_losses['optuna_loss']})')

    return losses 


def train_epoch(loader, model, criterion, optimizer, scaler, epoch_n):
    device = next(model.parameters()).device
    model.train()

    # loss tracking
    running_loss = 0.0          # the actual loss used in backpropagation
    running_optuna_loss = 0.0   # the loss passed to optuna (in case we are using custom criterions whose parameters are being optimized by optuna)
    running_losses = [0 for i in range(len(loader.dataset.targets[0]))] # each target's individual loss

    loop = tqdm(loader, desc=f"Epoch {epoch_n + 1}", leave=True, unit='batch')
    for inputs, targets in loop:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).float()

        # calculate the loss with AMP
        with torch.autocast(device_type=str(device), dtype=torch.float16):
            outputs = model(inputs)
            loss = criterion(outputs, targets.view_as(outputs)) # force targets to match the shape of predictions

        # Update running losses for tracking
        with torch.no_grad():
            base_criterion = criterion.criterion if hasattr(criterion, 'criterion') else criterion

            running_loss += loss.item()
            running_optuna_loss += base_criterion(outputs, targets.view_as(outputs)).item()

            for i in range(outputs.shape[1]):
                running_losses[i] += base_criterion(outputs[:, i], targets[:, i]).item()

        # backpropagation with AMP (uses the optimizer through the scaler)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True) # set_to none can modestly improve performance

        loop.set_postfix(loss=loss.item())

    return {
        "target_losses": [l / len(loader) for l in running_losses],
        "backprop_loss": running_loss / len(loader),
        "optuna_loss": running_optuna_loss / len(loader)
    }


def validate_epoch(loader, model, criterion):
    device = next(model.parameters()).device
    model.eval()

    # loss traking
    running_loss = 0.0
    running_optuna_loss = 0.0
    running_losses = [0 for i in range(len(loader.dataset.targets[0]))]

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device).float()
            
            # validation with AMP
            with torch.autocast(device_type=str(device), dtype=torch.float16):
                pred = model(X)
                y = y.view_as(pred)
            
                loss = criterion(pred, y)

                # loss tracking
                running_loss += loss.item()
                base_criterion = criterion.criterion if hasattr(criterion, 'criterion') else criterion
                running_optuna_loss += base_criterion(pred, y).item()
                for i in range(len(running_losses)):
                    running_losses[i] += base_criterion(pred[:, i], y[:, i]).item()
                
    return {
        "target_losses": [l / len(loader) for l in running_losses],
        "backprop_loss": running_loss / len(loader),
        "optuna_loss": running_optuna_loss / len(loader)
    }
