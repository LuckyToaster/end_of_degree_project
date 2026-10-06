from pathlib import Path
import torch, optuna, gc
from torch.utils.data import DataLoader
from torch import nn
from torchvision.transforms import v2

from src.dataset import FoodDataset
from src.constants import STUDIES_DIR, CSV_PATH
from src.helpers.models import freeze, unfreeze, get_Swin_V2_S
from src.helpers.ml import train_eval_loop, three_way_split, CustomCriterion

torch.cuda.empty_cache() if torch.cuda.is_available() else print('NO CUDA 🙉')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT = 'img_path'
TARGETS = ['fat_g', 'carb_g', 'prot_g', 'kcal']
SEED = 1
BS = 64

dataloader_args = dict(batch_size=BS, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
train_df, val_df, test_df = three_way_split(CSV_PATH, TARGETS, SEED)


def objective(trial):
    ALPHA = trial.suggest_float('custom_criterion_alpha', 0, 1)
    FE_LR = trial.suggest_float('fe_lr', 1e-4, 1e-2, log=True)
    FT_LR = trial.suggest_float('ft_lr', 1e-5, 1e-3, log=True)
    FE_WEIGHT_DECAY = trial.suggest_float('fe_weight_decay', 1e-4, 1e-1, log=True)
    FT_WEIGHT_DECAY = trial.suggest_float('ft_weight_decay', 1e-4, 1e-1, log=True)
    FE_EPOCHS = trial.suggest_int('fe_epochs', 5, 20)
    FT_EPOCHS = trial.suggest_int('ft_epochs', 20, 100)
    # LOSS = trial.suggest_categorical('loss', ['L1', 'MSE', 'Huber'])

    model, val_transforms = get_Swin_V2_S(verbose=False)
    train_transforms = v2.Compose([
        v2.RandomResizedCrop(size=256, scale=(0.8, 1.0), interpolation=v2.InterpolationMode.BICUBIC),
        v2.TrivialAugmentWide(interpolation=v2.InterpolationMode.BICUBIC), # flips, rotations and colorjitters
        # ImageNet and swin default transforms
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for param in model.parameters(): param.requires_grad = False    # freeze for feature extraction
    model.head = nn.Linear(model.head.in_features, len(TARGETS))    # adapt the head
    model = model.to(device)                                        # load to GPU

    train_loader = DataLoader(FoodDataset(train_df, train_transforms, INPUT, TARGETS), **dataloader_args)
    val_loader = DataLoader(FoodDataset(val_df, val_transforms, INPUT, TARGETS), **dataloader_args)

    # criterions = { 'L1': nn.L1Loss(), 'MSE': nn.MSELoss(), 'Huber': nn.HuberLoss() }
    criterion = CustomCriterion(nn.HuberLoss(), ALPHA, 1.0 - ALPHA)

    fe_optimizer, ft_optimizer = None, None
    try:
        # feature extraction (warm up)
        fe_optimizer = torch.optim.AdamW(model.head.parameters(), lr=FE_LR, weight_decay=FE_WEIGHT_DECAY, fused=True)
        train_eval_loop(model, FE_EPOCHS, train_loader, val_loader, criterion, fe_optimizer, device, trial)
        
        # clean up
        del fe_optimizer
        fe_optimizer = None
        gc.collect()
        torch.cuda.empty_cache()

        # fine tuning
        model.zero_grad(set_to_none=True)
        for param in model.parameters(): param.requires_grad = True    # unfreeze for training
        ft_optimizer = torch.optim.AdamW(model.parameters(), lr=FT_LR, weight_decay=FT_WEIGHT_DECAY, fused=True)
        ft_losses = train_eval_loop(model, FT_EPOCHS, train_loader, val_loader, criterion, ft_optimizer, device, trial, FE_EPOCHS)
        
        return ft_losses['val'][-1][-1] # last epoch avg loss

    finally:
        if fe_optimizer is not None: del fe_optimizer
        if ft_optimizer is not None: del ft_optimizer
        del train_loader, val_loader, model
        gc.collect()
        torch.cuda.empty_cache()



def main():
    Path(STUDIES_DIR).mkdir(exist_ok=True, parents=True)
    study = optuna.create_study(
        study_name='swin_ft_scl',
        storage=f'sqlite:///{STUDIES_DIR}/fine_tuning.db',
        direction='minimize',
        load_if_exists=True,
        pruner=optuna.pruners.HyperbandPruner()
    )
    study.optimize(objective, n_trials=100)
