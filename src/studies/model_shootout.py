import torch, optuna
import pandas as pd
from pathlib import Path

from torch.utils.data import DataLoader
from torch.nn import L1Loss
from torch.optim import AdamW

from src.constants import STUDIES_PATH, CSV_PATH
from src.dataset import FoodDataset
from src.helpers.ml import train_eval_loop, three_way_split
from src.helpers.models import get_EfficientNet_B3, get_EfficientNet_V2_S, get_MobileNet_V3_L, get_Swin_V2_S

INPUT = 'img_path'
TARGETS = ['fat_g', 'carb_g', 'prot_g']
SEED = 1
LR = 1e-4
EPOCHS = 25
BS = 32
OPTIM_WEIGHT_DECAY=0.01
MODEL_CONFIGS = {
    'EfficientNet_B3': get_EfficientNet_B3, 
    'EfficientNet_V2_S': get_EfficientNet_V2_S, 
    'MobileNet_V3_L': get_MobileNet_V3_L, 
    'Swin_V2_S': get_Swin_V2_S, 
}

torch.cuda.empty_cache() if torch.cuda.is_available() else print('NO CUDA 🙉')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_df, val_df, hidden_df = three_way_split(CSV_PATH, TARGETS, SEED)


def objective(trial, model_name):
    lr = trial.suggest_float('LR', 1e-5, 5e-4, log=True)

    model, transforms = MODEL_CONFIGS[model_name](feature_extraction=False, verbose=False)
    model = model.to(device)
    train_ds = FoodDataset(train_df, transform=transforms, input=INPUT, targets=TARGETS)
    val_ds = FoodDataset(val_df, transform=transforms, input=INPUT, targets=TARGETS)
    train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=BS, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=OPTIM_WEIGHT_DECAY)

    losses = train_eval_loop(model, EPOCHS, train_loader, val_loader, L1Loss(), optimizer, device, trial)
    return losses['val'][-1][-1] # last epoch average loss


def main():
    Path(STUDIES_PATH).mkdir(exist_ok=True, parents=True)
    
    for model_name in MODEL_CONFIGS.keys():
        study = optuna.create_study(
            study_name=f'shootout_{model_name}', 
            storage=f'sqlite:///{STUDIES_PATH}/model_shootout.db', 
            sampler=optuna.samplers.TPESampler(seed=SEED),
            direction='minimize',
            load_if_exists=True
        )
        study.optimize(lambda trial: objective(trial, model_name), n_trials=5)
