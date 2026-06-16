import torch
from torch.utils.data import DataLoader
from torch.nn import L1Loss
from torch.optim import AdamW
from src.constants import CSV_PATH
from src.dataset import FoodDataset
from src.helpers.ml import three_way_split
from src.helpers.models import get_Swin_V2_S

INPUT = 'img_path'
TARGETS = ['fat_g', 'carb_g', 'prot_g']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_df, val_df, _ = three_way_split(CSV_PATH, TARGETS, 1)

# Take just 100 samples to test
train_df = train_df.head(100)

model, transforms = get_Swin_V2_S(feature_extraction=False, verbose=False)
model = model.to(device)

train_ds = FoodDataset(train_df, transform=transforms, input=INPUT, targets=TARGETS)
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)

criterion = L1Loss()
optimizer = AdamW(model.parameters(), lr=1e-4)

model.train()
scaler = torch.amp.GradScaler('cuda')

for i, (inputs, targets) in enumerate(train_loader):
    inputs = inputs.to(device)
    targets = targets.to(device).float()
    
    optimizer.zero_grad()
    
    with torch.autocast(device_type=str(device), dtype=torch.float16):
        outputs = model(inputs)
        loss = criterion(outputs, targets.view_as(outputs))
        
    scaler.scale(loss).backward()
    
    # check gradients
    head_grad = model.head.weight.grad
    if head_grad is not None:
        print(f"Batch {i}: Loss = {loss.item():.4f}, Head grad norm = {head_grad.norm().item():.4f}")
    else:
        print(f"Batch {i}: Loss = {loss.item():.4f}, NO HEAD GRADIENTS!")
        
    scaler.step(optimizer)
    scaler.update()

    print(f"Outputs: {outputs.detach().cpu().numpy()[:2]}")
    print(f"Targets: {targets.detach().cpu().numpy()[:2]}")

    if i >= 5:
        break
