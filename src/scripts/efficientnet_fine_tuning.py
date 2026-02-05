from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.utils.data import DataLoader
from torch.nn import Linear, MSELoss, Sequential, Dropout
from torch.nn.init import xavier_uniform_
from torch.optim import AdamW
from torch import device, cuda
from sklearn.model_selection import train_test_split
import pandas as pd

from src.mmfood100k.dataset import MMFood100KDataset
from helpers import standardize, train_loop
from models.efficientnet import get_model

DEVICE = device("cuda" if cuda.is_available() else "cpu")
SEED = 1
EPOCHS = 5
LR = 1e-4
TARGETS = ['fat_g', 'carb_g', 'protein_g']


model, transforms = get_model()
model = model.to(DEVICE)

train_df, test_df = train_test_split(
        pd.read_csv('data/mm-food-100k/mm-food-100k.csv'), 
        test_size=0.1, 
        random_state=SEED
)

print(train_df[TARGETS].describe()) # lets see what might be going wrong during standardization

# STANDARDIZE TARGETS
train_df[TARGETS], test_df[TARGETS] = standardize(train_df[TARGETS], test_df[TARGETS])

train_ds = MMFood100KDataset(train_df, transform=transforms)
test_ds = MMFood100KDataset(test_df, transform=transforms)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=True, num_workers=4)


train_loop(
    model = model, 
    epochs = EPOCHS, 
    loader = train_loader, 
    criterion = MSELoss(), 
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01), 
    device = DEVICE
)
