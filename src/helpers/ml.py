from tqdm.asyncio import tqdm
from sklearn.preprocessing import StandardScaler
from torch.nn.utils import clip_grad_norm_
import torch
import matplotlib.pyplot as plt

__all__ = ['train_eval_loop', 'validate', 'standardize']


def train_eval_loop(model, epochs, train_loader, test_loader, criterion, optimizer, device):
    losses = {'train': [], 'val': []}
    for epoch in range(epochs): 
        train_avg_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_avg_loss = validate(test_loader, model, criterion, device)

        losses['train'].append(train_avg_loss)
        losses['val'].append(val_avg_loss) 
        print(f"Epoch {epoch+1} Complete - Train Loss: {losses['train'][-1]:.4f}, Val Loss: {losses['val'][-1]:.4f}")
    return losses


def train_epoch(model, train_loader, criterion, optimizer, device, epoch_n):
    model.train()
    running_loss = 0.0
    loop = tqdm(train_loader, desc=f"Epoch {epoch_n + 1}", leave=True, unit='batch')
    
    for inputs, targets in loop:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).float()
        optimizer.zero_grad()
        outputs = model(inputs)
        # loss = criterion(outputs, targets)
        loss = criterion(outputs, targets.view_as(outputs)) # force targets to match the shape of predictions
        loss.backward()
        # Added this to prevent exploding gradients in regression
        clip_grad_norm_(model.parameters(), max_norm=1.0) 
        optimizer.step()
        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    return running_loss / len(train_loader) # avg loss for that epoch


def validate(loader, model, loss_fn, device):
    model.eval()
    running_loss = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device).float()
            pred = model(X)
            running_loss += loss_fn(pred, y).item()
    return running_loss / len(loader)


def standardize(train_df, test_df):
    scaler = StandardScaler()
    scaler.set_output(transform="pandas")
    return scaler.fit_transform(train_df), scaler.transform(test_df)
