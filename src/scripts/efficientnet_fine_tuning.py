from torch.utils.data import DataLoader
from torch.nn import MSELoss 
from torch.optim import AdamW
from torch import device, cuda
from sklearn.model_selection import train_test_split
import pandas as pd

import torch

from src.mmfood100k.dataset import MMFood100KDataset
from src.helpers import standardize, train_loop, train
from src.models.efficientnet import get_model

DEVICE = device("cuda" if cuda.is_available() else "cpu")
SEED = 1
EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-4
TARGETS = ['fat_g', 'carb_g', 'protein_g']

if __name__ == '__main__':
    model, transforms = get_model()
    model = model.to(DEVICE)

    train_df, test_df = train_test_split(pd.read_csv('data/mm-food-100k/mm-food-100k.csv'), test_size=0.1, random_state=SEED)
    print(train_df[TARGETS].describe()) # lets see what might be going wrong during standardization

    # STANDARDIZE TARGETS
    train_df[TARGETS], test_df[TARGETS] = standardize(train_df[TARGETS], test_df[TARGETS])

    train_ds = MMFood100KDataset(train_df, transform=transforms, device=DEVICE)
    test_ds = MMFood100KDataset(test_df, transform=transforms, device=DEVICE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # del train_ds, test_ds, train_df, test_df

    losses = train_loop(
        model = model, 
        epochs = EPOCHS, 
        loader = train_loader, 
        criterion = MSELoss(), 
        optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01), 
        device = DEVICE
    )

    # train(
    #     dataloader = train_loader,
    #     model = model,
    #     loss_fn = MSELoss(),
    #     optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01),
    #     device = DEVICE
    # )
    torch.save(model.state_dict(), 'data')
