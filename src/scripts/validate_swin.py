import torch
from torch import nn
from torch.nn import HuberLoss, L1Loss, MSELoss
from torch.utils.data import DataLoader
import pandas as pd
import argparse

from src.dataset import FoodDataset
from src.constants import CSV_PATH
from src.helpers.models import get_Swin_V2_S
from src.helpers.ml import three_way_split, validate, get_scaler_on_train, get_predictions
import numpy as np

TARGETS = ['fat_g', 'carb_g', 'prot_g']
SEED = 1
BS = 64
SAVE_PATH = 'data/swin_validation.csv'


def validate_swin(checkpoint_path, loss_fn_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, hidden_df = three_way_split(CSV_PATH, TARGETS, SEED)
    
    scaler = get_scaler_on_train(CSV_PATH, TARGETS, SEED)
    
    model, transforms = get_Swin_V2_S(feature_extraction=True, verbose=False, modify_head=False)
    N_UNITS = 813
    DROPOUT = 0.2635
    model.head = nn.Sequential(
        nn.Linear(model.head.in_features, N_UNITS),
        nn.ReLU(),
        nn.Dropout(DROPOUT),
        nn.Linear(N_UNITS, 3)
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)

    # Setup dataloader for predictions
    loader = DataLoader(
        FoodDataset(hidden_df, transform=transforms, input='img_path', targets=TARGETS),
        batch_size=BS, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Calculate losses (using existing helper)
    losses = {'L1': L1Loss(), 'MSE': MSELoss(), 'Huber': HuberLoss()}
    val_losses = validate(loader, model, losses[loss_fn_name], device)
    print(f"Validation Losses: {val_losses}")

    preds, targets = get_predictions(loader, model, device)
    preds_grams = scaler.inverse_transform(preds.numpy())
    targets_grams = scaler.inverse_transform(targets.numpy())
    mae_grams = np.mean(np.abs(preds_grams - targets_grams), axis=0)
    
    return {
        'checkpoint': checkpoint_path,
        'fat_g_loss': val_losses[0],
        'carb_g_loss': val_losses[1],
        'prot_g_loss': val_losses[2],
        'total_loss': val_losses[3],
        'fat_mae_g': mae_grams[0],
        'carb_mae_g': mae_grams[1],
        'prot_mae_g': mae_grams[2]
    }
    # return results 
    #
    # pd.DataFrame([results]).to_csv(SAVE_PATH, index=False)


def main():
    parser = argparse.ArgumentParser(description="Validate Swin model on hidden data")
    parser.add_argument("checkpoint_path", type=str, help="Path to the model checkpoint")
    parser.add_argument("--loss_fn", type=str, default="Huber", choices=['L1', 'MSE', 'Huber'], help="Loss function used during training")
    args = parser.parse_args()
    results = validate_swin(args.checkpoint_path, args.loss_fn)
    print(f"Gram MAE: {results['fat_mae_g']:.2f}g fat, {results['carb_mae_g']:.2f}g carbs, {results['prot_mae_g']:.2f}g protein")
