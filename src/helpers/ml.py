from tqdm.asyncio import tqdm
from sklearn.preprocessing import StandardScaler
from torch.nn.utils import clip_grad_norm_
import torch

__all__ = ['train', 'test', 'standardize']


def train(model, epochs, loader, criterion, optimizer, device):
    model.train()
    epochs_losses = []
    for epoch in range(epochs): 
        running_loss = 0.0
        loop = tqdm(loader, desc=f"Epoch {epoch+1}", leave=True, unit='batch')
        
        for inputs, targets in loop:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).float()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            # Added this to prevent exploding gradients in regression
            clip_grad_norm_(model.parameters(), max_norm=1.0) 
            optimizer.step()
            
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        epochs_losses.append(running_loss / len(loader))
        print(f"Epoch {epoch+1} Complete - Avg Loss: {running_loss/len(loader):.4f}")
    return epochs_losses


def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def standardize(train_df, test_df):
    scaler = StandardScaler()
    scaler.set_output(transform="pandas")
    return scaler.fit_transform(train_df), scaler.transform(test_df)
