import torch
from src.helpers.models import get_Swin_V2_S
from torch.nn import L1Loss
model, transforms = get_Swin_V2_S(verbose=False)
model.eval()

# Dummy input
x = torch.randn(64, 3, 256, 256)
out = model(x)
print("Untrained model mean output:", out.mean(dim=0).detach().numpy())

criterion = L1Loss()
targets = torch.zeros_like(out)
print("Untrained model MAE against 0:", criterion(out, targets).item())
