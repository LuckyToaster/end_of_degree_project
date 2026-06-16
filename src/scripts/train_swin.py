import os
import gc
import torch
from torch import nn
from torch.nn import HuberLoss, L1Loss, MSELoss
from torch.utils.data import DataLoader
from pathlib import Path

from src.dataset import FoodDataset
from src.constants import CSV_PATH
from src.helpers.models import unfreeze, get_Swin_V2_S
from src.helpers.ml import train_eval_loop, three_way_split

# --- HYPERPARAMETERS ---
# Replace these with the best values from your Optuna study
N_UNITS = 813
DROPOUT = 0.2635
FE_LR = 7.867467714697837e-05
FT_LR = 9.399822412299438e-06
FE_WEIGHT_DECAY = 0.003979716794509852
FT_WEIGHT_DECAY = 0.004898850369631879
FE_EPOCHS = 9
FT_EPOCHS = 44
LOSS_FN = 'Huber' # 'L1', 'MSE', or 'Huber'

# --- TRAINING SETUP ---
INPUT = 'img_path'
TARGETS = ['fat_g', 'carb_g', 'prot_g']
SEED = 1
BS = 64
CHECKPOINTS_DIR = 'data/swin_v2_s'


def main():
    torch.cuda.empty_cache() if torch.cuda.is_available() else print('NO CUDA 🙉')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure checkpoint directory exists
    Path(CHECKPOINTS_DIR).mkdir(parents=True, exist_ok=True)

    # Prepare data
    train_df, val_df, hidden_df = three_way_split(CSV_PATH, TARGETS, SEED)

    # Initialize model
    model, transforms = get_Swin_V2_S(feature_extraction=True, verbose=True, modify_head=False)
    model.head = nn.Sequential(
        nn.Linear(model.head.in_features, N_UNITS),
        nn.ReLU(),
        nn.Dropout(DROPOUT),
        nn.Linear(N_UNITS, 3)
    )
    model = model.to(device)

    train_ds = FoodDataset(train_df, transform=transforms, input=INPUT, targets=TARGETS)
    val_ds = FoodDataset(val_df, transform=transforms, input=INPUT, targets=TARGETS)
    
    train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=BS, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    criterions = { 'L1': L1Loss(), 'MSE': MSELoss(), 'Huber': HuberLoss() }
    criterion = criterions[LOSS_FN]

    # --- Feature Extraction Phase ---
    print(f"Starting Feature Extraction for {FE_EPOCHS} epochs...")
    fe_optimizer = torch.optim.AdamW(model.head.parameters(), lr=FE_LR, weight_decay=FE_WEIGHT_DECAY, fused=True)
    
    train_eval_loop(
        model=model, 
        epochs=FE_EPOCHS, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        criterion=criterion, 
        optimizer=fe_optimizer, 
        device=device,
        save_dir=CHECKPOINTS_DIR,
        model_name="swin_fe"
    )
    
    del fe_optimizer
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()

    # --- Fine Tuning Phase ---
    print(f"Starting Fine-Tuning for {FT_EPOCHS} epochs...")
    unfreeze(model)
    ft_optimizer = torch.optim.AdamW([
        {'params': model.features.parameters(), 'lr': FT_LR}, 
        {'params': model.head.parameters(), 'lr': FE_LR}
    ], weight_decay=FT_WEIGHT_DECAY, fused=True)
    
    train_eval_loop(
        model=model, 
        epochs=FT_EPOCHS, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        criterion=criterion, 
        optimizer=ft_optimizer, 
        device=device, 
        starting_epoch=FE_EPOCHS,
        save_dir=CHECKPOINTS_DIR,
        model_name="swin_ft"
    )

    print("Training complete!")

if __name__ == '__main__':
    main()
