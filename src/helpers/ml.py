from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from optuna import TrialPruned
import torch
import pandas as pd
from pandas import DataFrame


def three_way_split(csv_path: str, targets: list[str], seed: int) -> tuple[DataFrame, DataFrame, DataFrame]:
    df = pd.read_csv(csv_path)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=seed)
    train_df[targets], test_df[targets] = standardize(train_df[targets], test_df[targets])
    val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=seed)
    return train_df, val_df, test_df


def standardize(train_df, test_df):
    scaler = StandardScaler()
    scaler.set_output(transform="pandas")
    return scaler.fit_transform(train_df), scaler.transform(test_df)


# embedding structural rules directly into the loss is called soft constraint optimization or physics informed optimization
class CustomCriterion(torch.nn.Module):
    def __init__(self, criterion_function, alpha=0.1, beta=0.5):
        super().__init__()
        self.criterion = criterion_function
        self.alpha = alpha
        self.beta = beta

    def forward(self, preds, targets): # preds (batch_size, 4) targets (fat, prot, carb
        mse_loss = self.criterion(preds, targets)

        pred_f = preds[:, 0]
        pred_p = preds[:, 1]
        pred_c = preds[:, 2]
        pred_cal = preds[:, 3]

        supposed_cal = (4 * pred_p) + (4 * pred_c) + (9 * pred_f) # TODO: or was it 8? 

        constraint_loss = self.criterion(supposed_cal, pred_cal)

        return (self.alpha * mse_loss) + (self.beta * constraint_loss)


def get_scaler_on_train(csv_path: str, targets: list[str], seed: int) -> StandardScaler:
    df = pd.read_csv(csv_path)
    train_df, _ = train_test_split(df, test_size=0.2, random_state=seed)
    scaler = StandardScaler()
    scaler.set_output(transform="pandas")
    scaler.fit(train_df[targets])
    return scaler


def get_predictions(loader, model, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            pred = model(X)
            all_preds.append(pred.cpu())
            all_targets.append(y)
    return torch.cat(all_preds), torch.cat(all_targets)


# get rid of the lasst redundant arguments
def train_eval_loop(model, epochs, train_loader, val_loader, criterion, optimizer, device, trial=None, starting_epoch=0, save_dir=None, model_name="model"):
    losses = {'train': [], 'val': []}
    scaler = torch.amp.GradScaler('cuda')
    
    if save_dir is not None:
        import os
        os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs): 
        actual_epoch = starting_epoch + epoch

        train_losses = train_epoch(train_loader, model, criterion, optimizer, device, actual_epoch, scaler)
        val_losses = validate(val_loader, model, criterion, device)

        losses['train'].append(train_losses)
        losses['val'].append(val_losses) 

        print(f'Epoch {actual_epoch+1} Complete - Train Losses {[round(l, 4) for l  in losses["train"][-1]]}, Val Losses: {[round(l, 4) for l in losses["val"][-1]]}')
        
        if save_dir is not None:
            save_path = os.path.join(save_dir, f"{model_name}_epoch_{actual_epoch+1}.pt")
            torch.save(model.state_dict(), save_path)
            print(f"Model checkpoint saved to {save_path}")

        if trial is not None:
            trial.report(val_losses[-1], actual_epoch)
            trial.set_user_attr("train_losses", trial.user_attrs.get("train_losses", []) + [train_losses])
            trial.set_user_attr("val_losses", trial.user_attrs.get("val_losses", []) + [val_losses])
            if trial.should_prune(): raise TrialPruned()

    return losses


def train_epoch(loader, model, criterion, optimizer, device, epoch_n, scaler):
    model.train()
    running_loss = 0.0
    running_raw_loss = 0.0
    running_losses = [0 for i in range(len(loader.dataset.targets[0]))]
    # running_losses = [0.0, 0.0, 0.0]

    loop = tqdm(loader, desc=f"Epoch {epoch_n + 1}", leave=True, unit='batch')
    for inputs, targets in loop:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).float()

        with torch.autocast(device_type=str(device), dtype=torch.float16):
            outputs = model(inputs)
            loss = criterion(outputs, targets.view_as(outputs)) # force targets to match the shape of predictions

        # Calculate individual losses (no gradients needed for tracking)
        with torch.no_grad():
            base_criterion = criterion.criterion if hasattr(criterion, 'criterion') else criterion
            raw_loss = base_criterion(outputs, targets.view_as(outputs))
            running_raw_loss += raw_loss.item()
            for i in range(outputs.shape[1]): # range(3)
                # Calculate loss for just the i-th column/output
                ind_loss = base_criterion(outputs[:, i], targets[:, i])
                running_losses[i] += ind_loss.item()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True) # set_to none can modestly improve performance

        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(loader)
    avg_raw_loss = running_raw_loss / len(loader)
    avg_losses = [l / len(loader) for l in running_losses]
    avg_losses.append(avg_loss)
    avg_losses.append(avg_raw_loss)
    return avg_losses 


def validate(loader, model, criterion, device):
    model.eval()
    running_loss = 0.0
    running_raw_loss = 0.0
    running_losses = [0 for i in range(len(loader.dataset.targets[0]))]
    # [0.0, 0.0, 0.0]

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device).float()
            
            with torch.autocast(device_type=str(device), dtype=torch.float16):
                pred = model(X)
                y = y.view_as(pred)
            
                loss = criterion(pred, y)
                running_loss += loss.item()
                
                base_criterion = criterion.criterion if hasattr(criterion, 'criterion') else criterion
                raw_loss = base_criterion(pred, y)
                running_raw_loss += raw_loss.item()
                
                for i in range(len(running_losses)):
                    ind_loss = base_criterion(pred[:, i], y[:, i])
                    running_losses[i] += ind_loss.item()
                
    avg_loss = running_loss / len(loader)
    avg_raw_loss = running_raw_loss / len(loader)
    avg_losses = [l / len(loader) for l in running_losses]
    avg_losses.append(avg_loss)
    avg_losses.append(avg_raw_loss)
    return avg_losses 
