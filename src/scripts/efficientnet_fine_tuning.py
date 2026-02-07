# from torch.utils.data import DataLoader
# from torch.nn import MSELoss, HuberLoss
# from torch.optim import AdamW
from sklearn.model_selection import train_test_split
import pandas as pd
import torch, json
# from torch import device, cuda
from src.mmfood100k.dataset import MMFood100KDataset
from src.helpers import standardize, train_loop, train
from src.models.efficientnet import get_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 1
EPOCHS = 1
BATCH_SIZE = 32
LR = 1e-4
TARGETS = ['fat_g', 'carb_g', 'protein_g']

if __name__ == '__main__':
    model, transforms = get_model()
    model = model.to(DEVICE)

    df = pd.read_csv('data/mm-food-100k/mm-food-100k.csv')
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=SEED)

    print(train_df[TARGETS].describe()) 
    train_df[TARGETS], test_df[TARGETS] = standardize(train_df[TARGETS], test_df[TARGETS])

    train_ds = MMFood100KDataset(train_df, transform=transforms, device=DEVICE)
    test_ds = MMFood100KDataset(test_df, transform=transforms, device=DEVICE)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=6)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=6)

    del df, train_df, test_df, train_ds, test_ds 

    losses = train_loop(
        model = model, 
        epochs = EPOCHS, 
        loader = train_loader, 
        criterion = torch.nn.HuberLoss(), 
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01), 
        device = DEVICE
    )

    torch.save(model.state_dict(), 'data/efnetb0_1epoch.pt')
    with open("data/efnetb0_1poch.json", "w") as f:
        json.dump({'losses per epoch': losses}, f, indent=4) 
