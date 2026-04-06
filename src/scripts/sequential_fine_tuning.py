import torch, optuna
import pandas as pd
from torch.utils.data import DataLoader
from torch import nn
from sklearn.model_selection import train_test_split
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

from src.mmfood100k.dataset import MMFood100KDataset
from src.helpers.models import freeze, unfreeze
from src.helpers.ml import standardize, train_eval_loop

INPUT = 'resized_img_path'
TARGETS = ['fat_g', 'carb_g', 'protein_g']
SEED = 1
BS = 128


def objective(trial):
    N_UNITS = trial.suggest_int('n_units', 256, 1024)
    DROPOUT = trial.suggest_float('dropout', 0.1, 0.5)
    FE_LR = trial.suggest_float('fe_lr', 1e-5, 1e-2, log=True)
    FT_LR = trial.suggest_float('ft_lr', 1e-6, 1e-4, log=True)
    FE_WEIGHT_DECAY = trial.suggest_float('fe_weight_decay', 1e-4, 1e-1, log=True)
    FT_WEIGHT_DECAY = trial.suggest_float('ft_weight_decay', 1e-4, 1e-1, log=True)
    FE_EPOCHS = trial.suggest_int('fe_epochs', 5, 20)
    FT_EPOCHS = trial.suggest_int('ft_epochs', 10, 50)
    LOSS = trial.suggest_categorical('loss', ['L1', 'MSE', 'Huber'])

    weights = MobileNet_V3_Large_Weights.DEFAULT
    transforms = weights.transforms()
    model = mobilenet_v3_large(weights=weights)
    freeze(model)
    # 2. Correctly adapt the head for regression
    # MobileNetV3 classifier input is 960 features
    in_features = model.classifier[0].in_features 
    # Replace the entire classifier to avoid shape mismatches
    model.classifier = nn.Sequential(
        nn.Linear(in_features, N_UNITS),
        nn.ReLU(),
        nn.Dropout(DROPOUT),
        nn.Linear(N_UNITS, 3)
    )

    model = model.to(device)

    train_ds = MMFood100KDataset(train_df, transform=transforms, input=INPUT, targets=TARGETS)
    test_ds = MMFood100KDataset(test_df, transform=transforms, input=INPUT, targets=TARGETS)
    train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=BS, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)

    if LOSS == 'L1': criterion = torch.nn.L1Loss()
    elif LOSS == 'MSE': criterion = torch.nn.MSELoss()
    else: criterion = torch.nn.HuberLoss()

    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=FE_LR, weight_decay=FE_WEIGHT_DECAY)
    
    train_eval_loop(
        model = model,
        epochs = FE_EPOCHS,
        train_loader = train_loader,
        test_loader = test_loader,
        criterion = criterion,
        optimizer = optimizer,
        device = device
    )

    unfreeze(model)

    optimizer = torch.optim.AdamW([
        {'params': model.features.parameters(), 'lr': FT_LR}, 
        {'params': model.classifier.parameters(), 'lr': FE_LR}
    ], weight_decay=FT_WEIGHT_DECAY)

    ft_results = train_eval_loop(
        model = model,
        epochs = FT_EPOCHS,
        train_loader = train_loader,
        test_loader = test_loader,
        criterion = criterion,
        optimizer = optimizer,
        device = device
    )
    
    # Return the best validation loss from the fine-tuning stage
    # ft_results['val'] is a list of lists: [epoch1_losses, epoch2_losses, ...]
    # Each epoch_losses is [fat_loss, carb_loss, protein_loss, total_avg_loss]
    last_epoch_avg_loss = ft_results['val'][-1][-1]
    return last_epoch_avg_loss


if __name__ == '__main__':
    torch.cuda.empty_cache() if torch.cuda.is_available() else print('NO CUDA 🙉')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv('data/mm-food-100k/mm-food-100k.csv')
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=SEED)
    train_df[TARGETS], test_df[TARGETS] = standardize(train_df[TARGETS], test_df[TARGETS])

    study = optuna.create_study(
        study_name='sequential_fine_tuning_v1', 
        storage='sqlite:///sequential_fine_tuning.db', 
        direction='minimize',
        load_if_exists=True
    )
    study.optimize(objective, n_trials=100)
