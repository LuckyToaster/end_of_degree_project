import torch, optuna
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from src.dataset import FoodDataset
from src.helpers.ml import standardize, train_eval_loop, lr_linear_scaling
from src.helpers.models import get_EfficientNet_B3, get_EfficientNet_V2_S, get_MobileNet_V3_L, get_Swin_V2_S 

INPUT = 'img_path'
TARGETS = ['fat_g', 'carb_g', 'prot_g']
SEED = 1
LR = 1e-3
EPOCHS = 10
BS = 32
MODEL_CONFIGS = {
    'EfficientNet_B3': get_EfficientNet_B3, 
    'EfficientNet_V2_S': get_EfficientNet_V2_S, 
    'MobileNet_V3_L': get_MobileNet_V3_L, 
    'Swin_V2_S': get_Swin_V2_S, 
}

torch.cuda.empty_cache() if torch.cuda.is_available() else print('NO CUDA 🙉')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df = pd.read_csv('data/food_dataset.csv')

train_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED)
train_df.loc[:, TARGETS], test_df.loc[:, TARGETS] = standardize(train_df[TARGETS], test_df[TARGETS])
val_df, hidden_df = train_test_split(test_df, test_size=0.5, random_state=SEED)


def objective(trial):
    model_name = trial.suggest_categorical('MODEL', list(MODEL_CONFIGS.keys()))
    model, transforms = MODEL_CONFIGS[model_name](feature_extraction=True, verbose=False)
    
    model = model.to(device)
    lr = lr_linear_scaling(LR, 32, BS)

    train_ds = FoodDataset(train_df, transform=transforms, input=INPUT, targets=TARGETS)
    val_ds = FoodDataset(val_df, transform=transforms, input=INPUT, targets=TARGETS)
    train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=BS, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)

    losses = train_eval_loop(
        model = model,
        epochs = EPOCHS,
        train_loader = train_loader,
        test_loader = val_loader,
        criterion = torch.nn.L1Loss(), # MAE
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01),
        device = device,
        trial = trial
    )
    
    last_epoch_avg_loss = losses['val'][-1][-1]
    return last_epoch_avg_loss


def main():
    path = 'data/studies'
    Path(path).mkdir(exist_ok=True, parents=True)
    search_space = { 'MODEL': list(MODEL_CONFIGS.keys()) }
    study = optuna.create_study(
        study_name='model_shootout', 
        storage=f'sqlite:///{path}/model_shootout.db', 
        sampler=optuna.samplers.GridSampler(search_space),
        direction='minimize',
        load_if_exists=True
    )
    study.optimize(objective, n_trials=4)
