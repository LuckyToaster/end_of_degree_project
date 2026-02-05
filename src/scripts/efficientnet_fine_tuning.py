from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.utils.data import DataLoader
from torch.nn import Linear, MSELoss
from torch.optim import AdamW
from torch import device, cuda
from sklearn.model_selection import train_test_split
import fireducks.pandas as pd
from tqdm import tqdm

from mmfood100k.mmfood100k_dataset import MMFood100KDataset
from helpers import standardize

device = device("cuda" if cuda.is_available() else "cpu")
SEED = 1
targets = ['fat_g', 'carb_g', 'protein_g']

weights = EfficientNet_B0_Weights.DEFAULT
preprocess = weights.transforms() # img_transformed = preprocess(img)
model = efficientnet_b0(weights=weights)
model = model.to(device)

df = pd.read_csv('data/mm-food-100k/mm-food-100k.csv')
train_df, test_df = train_test_split(df, test_size=0.1, random_state=SEED)
train_df[targets], test_df[targets] = standardize(train_df[targets], test_df[targets])

train_ds = MMFood100KDataset(train_df, transform=preprocess)
test_ds = MMFood100KDataset(test_df, transform=preprocess)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=True, num_workers=4)

in_features = model.classifier[1].in_features
model.classifier[1] = Linear(in_features, 3)


criterion = MSELoss() # or L1Loss() (MAE)
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)


model.train()
for epoch in range(5): 
    running_loss = 0.0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=True)
    
    for inputs, targets in loop:
        inputs = inputs.to(device)
        targets = targets.to(device).float()
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    print(f"Epoch {epoch+1} Complete - Avg Loss: {running_loss/len(train_loader):.4f}")
