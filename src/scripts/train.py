import argparse
import optuna
import torch
import sys
from pathlib import Path
from torch.utils.data import DataLoader

from src.constants import OPTUNA_DB_PATH, MODEL_CHECKPOINTS_DIR, CSV_PATH
from src.dataset import FoodDataset
from src.ml import train_eval_loop, three_way_split, standardize, dataloader_args
from src.models import get_Swin_V2_S, get_EfficientNet_B3, get_EfficientNet_V2_S, get_MobileNet_V3_L

# Mapping from string name to getter function
MODEL_REGISTRY = {
    'swin_v2_s': get_Swin_V2_S,
    'efficientnet_b3': get_EfficientNet_B3,
    'efficientnet_v2_s': get_EfficientNet_V2_S,
    'mobilenet_v3_l': get_MobileNet_V3_L
}

def parse_args():

def main():
    parser = argparse.ArgumentParser(description="Train a model based on Optuna study parameters.")
    parser.add_argument("study_name", help="The name of the Optuna study to pull parameters from.")
    parser.add_argument("model_name", choices=list(MODEL_REGISTRY.keys()), help="The model architecture to train.")
    parser.add_argument("-t", "--trial", default="best", help="The trial number to use, or 'best'. Defaults to 'best'.")
    args = parser.parse_args()
    
    # 1. Load Optuna Study & Trial
    print(f"Loading study: {args.study_name}")
    try:
        study = optuna.load_study(study_name=args.study_name, storage=OPTUNA_DB_PATH)
    except Exception as e:
        print(f"Error loading study: {e}")
        sys.exit(1)

    if args.trial.lower() == 'best':
        trial = study.best_trial
        print(f"Using Best Trial (Number: {trial.number})")
    else:
        try:
            trial_num = int(args.trial)
            trial = study.trials[trial_num]
            print(f"Using Trial Number: {trial_num}")
        except (ValueError, IndexError):
            print(f"Invalid trial number: {args.trial}")
            sys.exit(1)

    # 2. Extract configuration from User Attributes
    print("Extracting configuration from study user attributes...")
    try:
        targets = study.user_attrs['targets']
        seed = study.user_attrs.get('seed', 42)
        batch_size = study.user_attrs.get('batch_size', 64)
        fe_epochs = study.user_attrs.get('fe_epochs', 5)
        ft_epochs = study.user_attrs.get('ft_epochs', 30)
    except KeyError as e:
        print(f"Missing required user attribute in study: {e}")
        sys.exit(1)

    # 3. Setup Data
    print("Preparing dataset...")
    train_df, val_df, test_df = three_way_split(CSV_PATH, seed, stratify_target='kcal')
    train_df, val_df, test_df, means, stds = standardize(train_df, val_df, test_df, targets)
    
    # 4. Initialize Model
    print(f"Initializing model: {args.model_name} for {len(targets)} targets...")
    model_getter = MODEL_REGISTRY[args.model_name]
    
    # Handle the fact that swin returns 3 things, while others might currently return 2 based on models.py state
    getter_result = model_getter(n_targets=len(targets), verbose=True)
    if len(getter_result) == 3:
        model, train_transforms, val_transforms = getter_result
    else:
        model, val_transforms = getter_result
        train_transforms = val_transforms # Fallback if training transforms aren't returned

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loader = DataLoader(FoodDataset(train_df, train_transforms, 'img_path', targets), batch_size=batch_size, shuffle=True, **dataloader_args)
    val_loader = DataLoader(FoodDataset(val_df, val_transforms, 'img_path', targets), batch_size=batch_size, shuffle=False, **dataloader_args)

    # 5. Extract Hyperparameters from Trial
    print("Extracting hyperparameters from trial...")
    params = trial.params
    # Note: The exact keys here depend on your study script. These are common examples based on swin_ft_scl.py
    fe_lr = params.get('fe_lr', 1e-3)
    ft_lr = params.get('ft_lr', 1e-4)
    fe_wd = params.get('fe_weight_decay', 1e-2)
    ft_wd = params.get('ft_weight_decay', 1e-2)
    
    # Assuming Huber loss for simplicity; you might want to pull this from user_attrs if dynamic
    criterion = torch.nn.HuberLoss() 
    
    Path(MODEL_CHECKPOINTS_DIR).mkdir(parents=True, exist_ok=True)
    model_save_path = f"{MODEL_CHECKPOINTS_DIR}/{args.model_name}_{args.study_name}_trial_{trial.number}"

    # 6. Feature Extraction Phase (Head only)
    print(f"\n--- Starting Feature Extraction ({fe_epochs} Epochs) ---")
    for param in model.parameters(): param.requires_grad = False
    
    # Handle different model head attribute names (classifier vs head)
    head_params = model.head.parameters() if hasattr(model, 'head') else model.classifier.parameters()
    for param in head_params: param.requires_grad = True
        
    fe_optimizer = torch.optim.AdamW(head_params, lr=fe_lr, weight_decay=fe_wd, fused=True)
    
    train_eval_loop(model, train_loader, val_loader, fe_epochs, criterion, fe_optimizer, save_dir=MODEL_CHECKPOINTS_DIR, model_name=f"{args.model_name}_fe")
    
    # 7. Fine Tuning Phase (Full model)
    print(f"\n--- Starting Fine Tuning ({ft_epochs} Epochs) ---")
    for param in model.parameters(): param.requires_grad = True
    
    ft_optimizer = torch.optim.AdamW(model.parameters(), lr=ft_lr, weight_decay=ft_wd, fused=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(ft_optimizer, T_max=ft_epochs)
    
    train_eval_loop(model, train_loader, val_loader, ft_epochs, criterion, ft_optimizer, scheduler=scheduler, starting_epoch=fe_epochs, save_dir=MODEL_CHECKPOINTS_DIR, model_name=f"{args.model_name}_ft")
    
    print(f"\nTraining Complete! Checkpoints saved to {MODEL_CHECKPOINTS_DIR}")

if __name__ == "__main__":
    main()
