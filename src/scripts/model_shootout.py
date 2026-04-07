import torch, optuna
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from src.mmfood100k.dataset import MMFood100KDataset
from src.helpers.ml import standardize, train_eval_loop, lr_linear_scaling
from src.helpers.models import get_EfficientNet_B3, get_EfficientNet_V2_S, get_MobileNet_V3_L, get_Swin_V2_S 

INPUT = 'resized_img_path'
TARGETS = ['fat_g', 'carb_g', 'protein_g']
SEED = 1
LR = 1e-3
EPOCHS = 10
MODEL_CONFIGS = {
    'EfficientNet_B3': { 
        'func': get_EfficientNet_B3, 
        'bs': 32 
    },
    'EfficientNet_V2_S': { 
        'func': get_EfficientNet_V2_S, 
        'bs': 32 
    },
    'MobileNet_V3_L': { 
        'func': get_MobileNet_V3_L, 
        'bs': 256, 
    },
    'Swin_V2_S': { 
        'func': get_Swin_V2_S, 
        'bs': 32
    }
}


def objective(trial):
    model_name = trial.suggest_categorical('MODEL', list(MODEL_CONFIGS.keys()))
    model, transforms = MODEL_CONFIGS[model_name]['func'](feature_extraction=True, verbose=False)
    
    model = model.to(device)
    bs = MODEL_CONFIGS[model_name]['bs']
    lr = lr_linear_scaling(LR, 32, bs)

    train_ds = MMFood100KDataset(train_df, transform=transforms, input=INPUT, targets=TARGETS)
    test_ds = MMFood100KDataset(test_df, transform=transforms, input=INPUT, targets=TARGETS)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)

    losses = train_eval_loop(
        model = model,
        epochs = EPOCHS,
        train_loader = train_loader,
        test_loader = test_loader,
        criterion = torch.nn.L1Loss(), # MAE
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01),
        device = device
    )
    
    last_epoch_avg_loss = losses['val'][-1][-1]
    return last_epoch_avg_loss


if __name__ == '__main__':
    torch.cuda.empty_cache() if torch.cuda.is_available() else print('NO CUDA 🙉')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    df = pd.read_csv('data/mm-food-100k/mm-food-100k.csv')
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=SEED)
    train_df[TARGETS], test_df[TARGETS] = standardize(train_df[TARGETS], test_df[TARGETS])

    search_space = { 'MODEL': list(MODEL_CONFIGS.keys()) }
    study = optuna.create_study(
        study_name='model_shootout_v1', 
        storage='sqlite:///model_shootout.db', 
        sampler=optuna.samplers.GridSampler(search_space),
        direction='minimize',
        load_if_exists=True
    )
    study.optimize(objective, n_trials=10)
