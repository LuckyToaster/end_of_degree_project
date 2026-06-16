import torch
from torch.utils.data import DataLoader
from torch.nn import L1Loss
from torch.optim import AdamW
from src.constants import CSV_PATH
from src.dataset import FoodDataset
from src.helpers.ml import three_way_split
from src.helpers.models import get_EfficientNet_B3
import sys

torch.cuda.empty_cache()

INPUT = 'img_path'
TARGETS = ['fat_g', 'carb_g', 'prot_g']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_df, val_df, _ = three_way_split(CSV_PATH, TARGETS, 1)
train_df = train_df.head(100)

model, transforms = get_EfficientNet_B3(feature_extraction=False, verbose=False)
model = model.to(device)

train_ds = FoodDataset(train_df, transform=transforms, input=INPUT, targets=TARGETS)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)

criterion = L1Loss()
optimizer = AdamW(model.parameters(), lr=1e-4)

model.train()
scaler = torch.amp.GradScaler('cuda')

print(f"Initial Scale: {scaler.get_scale()}")

for i, (inputs, targets) in enumerate(train_loader):
    inputs = inputs.to(device)
    targets = targets.to(device).float()
    
    optimizer.zero_grad()
    
    with torch.autocast(device_type=str(device), dtype=torch.float16):
        outputs = model(inputs)
        loss = criterion(outputs, targets.view_as(outputs))
        
    scaler.scale(loss).backward()
    
    scale_before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    scale_after = scaler.get_scale()
    
    # Just grab one weight from the linear head
    head_weight = model.classifier[1].weight.detach().cpu().numpy()[0][0]
    
    print(f"Batch {i}: Loss = {loss.item():.4f}, Scale before = {scale_before}, Scale after = {scale_after}, Head weight = {head_weight:.6f}")

    del loss, outputs
    if i >= 10:
        break
