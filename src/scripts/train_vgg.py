from torchvision.transforms.v2 import Resize, Compose, ToImage, ToDtype
from torch.utils.data import DataLoader
from torch.nn import MSELoss 
from torch.optim import AdamW
from torch import device, cuda
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
from src.mmfood100k.dataset import MMFood100KDataset
from src.helpers import standardize, train
from src.models.vgg import VGG
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


DEVICE = device("cuda" if cuda.is_available() else "cpu")
SEED = 1
EPOCHS = 5
LR = 1e-4
TARGETS = ['fat_g', 'carb_g', 'protein_g']

transform_pipeline = Compose([
    ToImage(),                            # Converts PIL to Tensor/TVTensor
    ToDtype(torch.float32, scale=True),   # Scales pixels to [0, 1]
    Resize((224, 224), antialias=True)    # Performs the resize
])


df = pd.read_csv('data/mm-food-100k/mm-food-100k.csv')
train_df, test_df = train_test_split(df, test_size=0.1, random_state=SEED)
train_df[TARGETS], test_df[TARGETS] = standardize(train_df[TARGETS], test_df[TARGETS])
train_ds = MMFood100KDataset(train_df, transform=transform_pipeline)
test_ds = MMFood100KDataset(test_df, transform=transform_pipeline)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=True)
del df, train_df, test_df, train_ds, test_ds

model = VGG(n_targets=3).to(DEVICE)

train(
    dataloader = train_loader,
    model = model,
    loss_fn = MSELoss(),
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01),
    device = DEVICE
)
