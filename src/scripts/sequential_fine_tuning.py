import torch, optuna
import pandas as pd
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import HuberLoss, L1Loss, MSELoss
from sklearn.model_selection import train_test_split
from torchvision.models import swin_v2_s, Swin_V2_S_Weights
from src.mmfood100k.dataset import MMFood100KDataset
from src.helpers.models import freeze, unfreeze
from src.helpers.ml import standardize, train_eval_loop
import gc

INPUT = 'resized_img_path'
TARGETS = ['fat_g', 'carb_g', 'protein_g']
SEED = 1
BS = 32

def get_swin_v2_s_pretrained():
    weights = Swin_V2_S_Weights.DEFAULT
    transforms = weights.transforms()
    model = swin_v2_s(weights=weights)
    return model, transforms


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


    model, transforms = get_swin_v2_s_pretrained()
    freeze(model)
    model.head = nn.Sequential(
        nn.Linear(model.head.in_features, N_UNITS),
        nn.ReLU(),
        nn.Dropout(DROPOUT),
        nn.Linear(N_UNITS, 3)
    )
    model = model.to(device)

    train_ds = MMFood100KDataset(train_df, transform=transforms, input=INPUT, targets=TARGETS)
    test_ds = MMFood100KDataset(test_df, transform=transforms, input=INPUT, targets=TARGETS)
    train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=BS, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)

    try:
        criterions = { 'L1': L1Loss(), 'MSE': MSELoss(), 'Huber': HuberLoss() }

        train_eval_loop(
            model = model,
            epochs = FE_EPOCHS,
            train_loader = train_loader,
            test_loader = test_loader,
            criterion = criterions[LOSS],
            device = device,
            trial = trial,
            starting_epoch = 0,
            optimizer = torch.optim.AdamW(model.head.parameters(), lr=FE_LR, weight_decay=FE_WEIGHT_DECAY),
        )

        unfreeze(model)

        ft_results = train_eval_loop(
            model = model,
            epochs = FT_EPOCHS,
            train_loader = train_loader,
            test_loader = test_loader,
            criterion = criterions[LOSS],
            device = device,
            trial = trial,
            starting_epoch = FE_EPOCHS,
            optimizer = torch.optim.AdamW(
                [
                    {'params': model.features.parameters(), 'lr': FT_LR}, 
                    {'params': model.head.parameters(), 'lr': FE_LR}
                ], 
                weight_decay=FT_WEIGHT_DECAY
            )
        )

        last_epoch_avg_loss = ft_results['val'][-1][-1]
        return last_epoch_avg_loss

    finally:
        # clean shit up to avoid linux killing process due to OOM 
        if 'model' in locals(): del model
        if 'train_ds' in locals(): del train_ds
        if 'test_ds' in locals(): del test_ds
        if 'train_loader' in locals(): del train_loader
        if 'test_loader' in locals(): del test_loader
        gc.collect()
        torch.cuda.empty_cache()



if __name__ == '__main__':
    torch.cuda.empty_cache() if torch.cuda.is_available() else print('NO CUDA 🙉')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv('data/mm-food-100k/mm-food-100k.csv')
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=SEED)
    train_df[TARGETS], test_df[TARGETS] = standardize(train_df[TARGETS], test_df[TARGETS])

    study = optuna.create_study(
        study_name='sequential_fine_tuning', 
        storage='sqlite:///sequential_fine_tuning.db', 
        direction='minimize',
        load_if_exists=True,
        pruner=optuna.pruners.HyperbandPruner()
    )
    study.optimize(objective, n_trials=500)
