from tqdm.asyncio import tqdm
from sklearn.preprocessing import StandardScaler
from torch.nn.utils import clip_grad_norm_
import torch

__all__ = ['train_eval_loop', 'validate', 'standardize', 'lr_linear_scaling']


lr_linear_scaling = lambda lr, old_bs, new_bs: lr * (new_bs / old_bs)


def train_eval_loop(model, epochs, train_loader, test_loader, criterion, optimizer, device):
    losses = {'train': [], 'val': []}
    for epoch in range(epochs): 
        train_losses = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_losses = validate(test_loader, model, criterion, device)

        losses['train'].append(train_losses)
        losses['val'].append(val_losses) 

        res_train = [round(l, 4) for l  in losses['train'][-1]]
        res_val = [round(l, 4) for l in losses['val'][-1]]
        print(f'Epoch {epoch+1} Complete - Train Losses {res_train}, Val Losses: {res_val}')
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


def standardize(train_df, test_df):
    scaler = StandardScaler()
    scaler.set_output(transform="pandas")
    return scaler.fit_transform(train_df), scaler.transform(test_df)
