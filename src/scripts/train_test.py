from argparse import ArgumentParser
import optuna
import sys, gc

from pathlib import Path

import torch
# from torch import device

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import HuberLoss
from torch.utils.data import DataLoader

from sklearn.metrics import mean_absolute_error

from src.constants import OPTUNA_DB_PATH, MODEL_CHECKPOINTS_DIR, CSV_PATH, SEED, INPUT
from src.dataset import FoodDataset
from src.ml import train_eval_loop, three_way_split, standardize, destandardize, dataloader_args
from src.models import get_Swin_V2_S, get_EfficientNet_B3, get_EfficientNet_V2_S, get_MobileNet_V3_L

TARGETS = ['fat_g', 'carb_g', 'prot_g', 'kcal']
BS = 16
DEVICE = torch.device('cuda')
MODELS = {
    'swin_v2_s': get_Swin_V2_S,
    'efficientnet_b3': get_EfficientNet_B3,
    'efficientnet_v2_s': get_EfficientNet_V2_S,
    'mobilenet_v3_l': get_MobileNet_V3_L
}

fe_epochs = 2
ft_epochs = 2


def main():
    parser = ArgumentParser(description="Train a model based on Optuna study parameters.")
    parser.add_argument("study_name", help="The name of the Optuna study to pull parameters from.")
    parser.add_argument("model_name", choices=list(MODELS.keys()), help="The model architecture to train.")
    parser.add_argument("-t", "--trial", default="best", help="The trial number to use, or 'best'. Defaults to 'best'.")
    args = parser.parse_args()

    train_df, val_df, test_df = three_way_split(CSV_PATH, SEED, stratify_target='kcal')
    train_df, val_df, test_df, means, stds = standardize(train_df, val_df, test_df, TARGETS)

    # guard, resolve, execute
    try:
        study = optuna.load_study(study_name=args.study_name, storage=OPTUNA_DB_PATH)
    except: raise ValueError(f'Study name \'{args.study_name}\' does not exist')

    if args.trial >= len(study.trials):
        raise ValueError(f'There are {len(study.trials)} trials in {args.study_name}. The largest --trial value permitted is {len(study.trials) - 1}')

    # trial
    trial = study.trials[args.trial] if type(args.trial) == int else study.best_trial
    params = trial.params

    # model
    model, train_transforms, val_transforms = MODELS[args.model_name]()
    for param in model.parameters(): param.requires_grad = False    # freeze for feature extraction
    model = model.to(DEVICE)                                        # load to GPU

    # dataloaders
    train_loader = DataLoader(FoodDataset(train_df, train_transforms, INPUT, TARGETS), batch_size=BS, shuffle=True, **dataloader_args)
    val_loader = DataLoader(FoodDataset(val_df, val_transforms, INPUT, TARGETS), batch_size=BS, shuffle=False, **dataloader_args)

    # feature extraction
    optimizer = AdamW(model.head.parameters(), lr=params['fe_lr'], weight_decay=params['fe_weight_decay'], fused=True)
    # train_eval_loop(model, train_loader, val_loader, params['fe_epochs'], HuberLoss(), optimizer)
    train_eval_loop(model, train_loader, val_loader, fe_epochs, HuberLoss(), optimizer)
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()

    # fine tuning
    model.zero_grad(set_to_none=True)
    for param in model.parameters(): param.requires_grad = True    # unfreeze for training

    optimizer = AdamW(model.parameters(), lr=params['ft_lr'], weight_decay=params['ft_weight_decay'], fused=True)
    # scheduler = CosineAnnealingLR(optimizer, T_max=params['ft_epochs'])
    scheduler = CosineAnnealingLR(optimizer, T_max=ft_epochs)

    # ft_losses = train_eval_loop(model, train_loader, val_loader, params['ft_epochs'], HuberLoss(), optimizer, scheduler)
    ft_losses = train_eval_loop(model, train_loader, val_loader, ft_epochs, HuberLoss(), optimizer, scheduler)
    del optimizer, train_loader, val_loader
    gc.collect()
    torch.cuda.empty_cache()

    # test
    test_loader = DataLoader(FoodDataset(test_df, val_transforms, INPUT, TARGETS), batch_size=BS, shuffle=False, **dataloader_args)
    for X, y in test_loader:
        X, y = X.to(DEVICE), y.to(DEVICE).float()
        with torch.autocast(device_type=str(DEVICE), dtype=torch.float16):
            pred = model(X)
            y = y.view_as(pred)

            preds_unscaled = destandardize(pred, means, stds)
            y_unscaled = destandardize(y, means, stds)
            mae = mean_absolute_error(y_unscaled, preds_unscaled)
            print(mae)

