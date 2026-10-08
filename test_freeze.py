import torch
import torch.nn as nn
from torchvision.models import swin_v2_s

model = swin_v2_s()
model.head = nn.Linear(model.head.in_features, 3)

for param in model.parameters():
    param.requires_grad = False

opt = torch.optim.AdamW(model.head.parameters(), lr=0.01)

loss = model(torch.randn(1, 3, 256, 256)).sum()
try:
    loss.backward()
    print("BACKWARD SUCCESS")
except Exception as e:
    print(f"BACKWARD FAILED: {e}")
