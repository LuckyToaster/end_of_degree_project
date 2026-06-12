from tqdm.asyncio import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from optuna import TrialPruned
import torch
import pandas as pd
from pandas import DataFrame


def three_way_split(csv_path: str, targets: list[str], seed: int) -> (DataFrame, DataFrame, DataFrame):
    df = pd.read_csv(csv_path)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=seed)
    train_df[targets], test_df[targets] = standardize(train_df[targets], test_df[targets])
    val_df, hidden_df = train_test_split(test_df, test_size=0.5, random_state=seed)
    return train_df, val_df, hidden_df


def standardize(train_df, test_df):
    scaler = StandardScaler()
    scaler.set_output(transform="pandas")
    return scaler.fit_transform(train_df), scaler.transform(test_df)


def train_eval_loop(model, epochs, train_loader, val_loader, criterion, optimizer, device, trial=None, starting_epoch=0):
    losses = {'train': [], 'val': []}
    for epoch in range(epochs): 
        actual_epoch = starting_epoch + epoch
        train_losses = train_epoch(model, train_loader, criterion, optimizer, device, actual_epoch)
        val_losses = validate(val_loader, model, criterion, device)

        losses['train'].append(train_losses)
        losses['val'].append(val_losses) 

        print(f'Epoch {actual_epoch+1} Complete - Train Losses {[round(l, 4) for l  in losses['train'][-1]]}, Val Losses: {[round(l, 4) for l in losses['val'][-1]]}')

        if trial is not None:
            trial.report(val_losses[-1], actual_epoch)
            trial.set_user_attr("train_losses", trial.user_attrs.get("train_losses", []) + [train_losses])
            trial.set_user_attr("val_losses", trial.user_attrs.get("val_losses", []) + [val_losses])
            if trial.should_prune(): raise TrialPruned()

    return losses


def train_epoch(model, train_loader, criterion, optimizer, device, epoch_n):
    model.train()
    running_loss = 0.0
    running_losses = [0.0, 0.0, 0.0]

    loop = tqdm(train_loader, desc=f"Epoch {epoch_n + 1}", leave=True, unit='batch')
    for inputs, targets in loop:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).float()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.view_as(outputs)) # force targets to match the shape of predictions

        # Calculate individual losses (no gradients needed for tracking)
        with torch.no_grad():
            for i in range(3):
                # Calculate loss for just the i-th column/output
                ind_loss = criterion(outputs[:, i], targets[:, i])
                running_losses[i] += ind_loss.item()

        loss.backward()
        # Added this to prevent exploding gradients in regression
        # clip_grad_norm_(model.parameters(), max_norm=1.0) 
        optimizer.step()
        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(train_loader)
    avg_losses = [l / len(train_loader) for l in running_losses]
    avg_losses.append(avg_loss)
    return avg_losses 


def validate(loader, model, criterion, device):
    model.eval()
    running_loss = 0.0
    running_losses = [0.0, 0.0, 0.0]

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device).float()
            pred = model(X)
            y = y.view_as(pred)
            
            loss = criterion(pred, y)
            running_loss += loss.item()
            
            for i in range(3):
                ind_loss = criterion(pred[:, i], y[:, i])
                running_losses[i] += ind_loss.item()
                
    avg_loss = running_loss / len(loader)
    avg_losses = [l / len(loader) for l in running_losses]
    avg_losses.append(avg_loss)
    return avg_losses 


