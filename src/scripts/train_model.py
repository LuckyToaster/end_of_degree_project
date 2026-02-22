import pandas as pd
import torch, json
from datetime import datetime as dt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.mmfood100k.dataset import MMFood100KDataset
from src.helpers.ml import standardize, train_eval_loop
# from src.models import get_EfficientNet_V2_S
# from src.models import get_EfficientNet_B3
from src.models import get_EfficientNet_B0

torch.cuda.empty_cache() if torch.cuda.is_available() else print('NO CUDA 🙉')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 1
EPOCHS = 100
BATCH_SIZE = 32
LR = 1e-4
TARGETS = ['fat_g', 'carb_g', 'protein_g']


if __name__ == '__main__':
    model, transforms = get_EfficientNet_B0()
    model = model.to(DEVICE)

    df = pd.read_csv('data/mm-food-100k/mm-food-100k.csv')
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=SEED)
    print(train_df[TARGETS].describe()) 
    train_df[TARGETS], test_df[TARGETS] = standardize(train_df[TARGETS], test_df[TARGETS])
    train_ds = MMFood100KDataset(train_df, transform=transforms, input='resized_img_path')
    test_ds = MMFood100KDataset(test_df, transform=transforms, input='resized_img_path')
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    del df, train_df, test_df, train_ds, test_ds 

    losses = train_eval_loop(
        model = model,
        epochs = EPOCHS,
        train_loader = train_loader,
        test_loader = test_loader,
        criterion = torch.nn.L1Loss(), 
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01),
        device = DEVICE
    )

    # SAVE
    filename = str(dt.now()).split('.')[0]
    torch.save(model.state_dict(), f'data/{filename}.pt')
    config = {
        'model': 'efficient_net_b3',
        'weights_file': filename,
        'seed': SEED,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LR,
        'targets': TARGETS,
        'loss': 'L1/MAE',
        'optimizer': 'adamw',
        'losses_epochs': losses 
    }
    with open(f'data/{filename}.json', 'w') as f:
        json.dump(config, f, indent=4) 

