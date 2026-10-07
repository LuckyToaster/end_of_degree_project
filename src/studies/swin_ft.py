from pathlib import Path
import torch, optuna, gc
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import HuberLoss, L1Loss, MSELoss

from src.dataset import FoodDataset
from src.constants import STUDIES_DIR, CSV_PATH
from src.helpers.models import freeze, unfreeze, get_Swin_V2_S
from src.helpers.ml import train_eval_loop, three_way_split

torch.cuda.empty_cache() if torch.cuda.is_available() else print('NO CUDA 🙉')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT = 'img_path'
TARGETS = ['fat_g', 'carb_g', 'prot_g']
SEED = 1
BS = 64

train_df, val_df, test_df = three_way_split(CSV_PATH, TARGETS, SEED)


def objective(trial):
    # N_UNITS = trial.suggest_categorical('n_units', [256, 512, 768, 1024])
    # DROPOUT = trial.suggest_float('dropout', 0.1, 0.5)
    FE_LR = trial.suggest_float('fe_lr', 1e-5, 1e-2, log=True)
    FT_LR = trial.suggest_float('ft_lr', 1e-6, 1e-3, log=True)
    FE_WEIGHT_DECAY = trial.suggest_float('fe_weight_decay', 1e-4, 1e-1, log=True)
    FT_WEIGHT_DECAY = trial.suggest_float('ft_weight_decay', 1e-4, 1e-1, log=True)
    FE_EPOCHS = trial.suggest_int('fe_epochs', 5, 25)
    FT_EPOCHS = trial.suggest_int('ft_epochs', 20, 120)
    LOSS = trial.suggest_categorical('loss', ['L1', 'MSE', 'Huber'])

    # model, transforms = get_Swin_V2_S(feature_extraction=True, verbose=False, modify_head=False)
    model, transforms = get_Swin_V2_S(verbose=False)
    freeze(model) 
    model.head = nn.Linear(model.head.in_features, 3)
    # model.head = nn.Sequential(
        # nn.Linear(model.head.in_features, N_UNITS),
        # nn.ReLU(),
        # nn.Dropout(DROPOUT),
    #     nn.Linear(N_UNITS, 3)
    # )
    model = model.to(device)

    train_ds = FoodDataset(train_df, transform=transforms, input=INPUT, targets=TARGETS)
    val_ds = FoodDataset(val_df, transform=transforms, input=INPUT, targets=TARGETS)
    train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=BS, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)

    criterions = { 'L1': L1Loss(), 'MSE': MSELoss(), 'Huber': HuberLoss() }
    fe_optimizer = None
    ft_optimizer = None

    try:
        fe_optimizer = torch.optim.AdamW(model.head.parameters(), lr=FE_LR, weight_decay=FE_WEIGHT_DECAY, fused=True)
        train_eval_loop(model, FE_EPOCHS, train_loader, val_loader, criterions[LOSS], fe_optimizer, device, trial)
        
        del fe_optimizer
        fe_optimizer = None
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()

        unfreeze(model)
        ft_optimizer = torch.optim.AdamW(model.parameters(), lr=FT_LR, weight_decay=FT_WEIGHT_DECAY, fused=True)
        ft_losses = train_eval_loop(model, FT_EPOCHS, train_loader, val_loader, criterions[LOSS], ft_optimizer, device, trial, FE_EPOCHS)
        
        return ft_losses['val'][-1][-1] # last epoch avg loss

    finally:
        if fe_optimizer is not None: del fe_optimizer
        if ft_optimizer is not None: del ft_optimizer
        del train_loader, val_loader, model, train_ds, val_ds
        gc.collect()
        torch.cuda.empty_cache()



def main():
    Path(STUDIES_DIR).mkdir(exist_ok=True, parents=True)
    study = optuna.create_study(
        study_name='seq_ft_october_no_hidden', 
        storage=f'sqlite:///{STUDIES_DIR}/sequential_fine_tuning.db',
        direction='minimize',
        load_if_exists=True,
        pruner=optuna.pruners.HyperbandPruner()
    )
    study.optimize(objective, n_trials=100)
