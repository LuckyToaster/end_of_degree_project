import torch, optuna, gc
from torch.utils.data import DataLoader
from torch import nn
from torchvision.transforms import v2

from src.dataset import FoodDataset
from src.constants import OPTUNA_DB_PATH, CSV_PATH
from src.helpers.models import get_Swin_V2_S
from src.helpers.ml import train_eval_loop, three_way_split, standardize, dataloader_args


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


INPUT = 'img_path'
TARGETS = ['fat_g', 'carb_g', 'prot_g', 'kcal']
SEED = 1
BS = 64
FE_EPOCHS = 5
FT_EPOCHS = 50

torch.cuda.empty_cache() if torch.cuda.is_available() else print('NO CUDA 🙉')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_df, val_df, test_df = three_way_split(CSV_PATH, SEED, stratify_target='kcal')
train_df, val_df, test_df, means, stds = standardize(train_df, val_df, test_df, TARGETS)


def objective(trial):
    ALPHA = trial.suggest_float('custom_criterion_alpha', 0, 1)
    FE_LR = trial.suggest_float('fe_lr', 1e-4, 1e-2, log=True)
    FT_LR = trial.suggest_float('ft_lr', 1e-5, 1e-3, log=True)
    FE_WEIGHT_DECAY = trial.suggest_float('fe_weight_decay', 1e-4, 1e-1, log=True)
    FT_WEIGHT_DECAY = trial.suggest_float('ft_weight_decay', 1e-4, 1e-1, log=True)

    model, train_transforms, val_transforms = get_Swin_V2_S(verbose=False)

    for param in model.parameters(): param.requires_grad = False    # freeze for feature extraction
    model = model.to(device)                                        # load to GPU

    train_loader = DataLoader(FoodDataset(train_df, train_transforms, INPUT, TARGETS), batch_size=BS, shuffle=True, **dataloader_args)
    val_loader = DataLoader(FoodDataset(val_df, val_transforms, INPUT, TARGETS), batch_size=BS, shuffle=False, **dataloader_args)

    criterion = CustomCriterion(nn.HuberLoss(), ALPHA, 1.0 - ALPHA)
    fe_optimizer, optimizer = None, None
    try:
        # feature extraction (warm up)
        fe_optimizer = torch.optim.AdamW(model.head.parameters(), lr=FE_LR, weight_decay=FE_WEIGHT_DECAY, fused=True)
        train_eval_loop(model, train_loader, val_loader, FE_EPOCHS, criterion, fe_optimizer, trial=trial)
        
        # clean up
        del fe_optimizer
        fe_optimizer = None
        gc.collect()
        torch.cuda.empty_cache()

        # fine tuning
        model.zero_grad(set_to_none=True)
        for param in model.parameters(): param.requires_grad = True    # unfreeze for training
        optimizer = torch.optim.AdamW(model.parameters(), lr=FT_LR, weight_decay=FT_WEIGHT_DECAY, fused=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FT_EPOCHS)
        ft_losses = train_eval_loop(model, train_loader, val_loader, FT_EPOCHS, criterion, optimizer, scheduler=scheduler, trial=trial, starting_epoch=FE_EPOCHS)
        
        return ft_losses['val'][-1]['optuna_loss'] 

    finally:
        if fe_optimizer is not None: del fe_optimizer
        if optimizer is not None: del optimizer
        if scheduler is not None: del scheduler
        del train_loader, val_loader, model
        gc.collect()
        torch.cuda.empty_cache()


def main():
    study = optuna.create_study(
        study_name='swin_ft_scl',
        storage=OPTUNA_DB_PATH,
        direction='minimize',
        load_if_exists=True,
        pruner=optuna.pruners.HyperbandPruner(min_resource=FE_EPOCHS, max_resource=FT_EPOCHS)
    )

    study.set_user_attr('targets', TARGETS)
    study.set_user_attr('seed', SEED)
    study.set_user_attr('batch_size', BS)
    study.set_user_attr('fe_epochs', FE_EPOCHS)
    study.set_user_attr('ft_epochs', FT_EPOCHS)
    study.set_user_attr('target_means', means)
    study.set_user_attr('target_stds', stds)

    study.optimize(objective, n_trials=200)


