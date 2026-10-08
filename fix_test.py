import torch
import torch.nn as nn
from torchvision.models import swin_v2_s

# HOW `models.py` works right now
model = swin_v2_s()
model.head = nn.Linear(model.head.in_features, 3)

# HOW `train_test.py` works right now
for param in model.parameters(): param.requires_grad = False
opt = torch.optim.AdamW(model.head.parameters(), lr=0.01)
print("Is head frozen before backward?", not list(model.head.parameters())[0].requires_grad)

# HOW `swin_ft_scl.py` works right now
for param in model.parameters(): param.requires_grad = False
opt_scl = torch.optim.AdamW(model.head.parameters(), lr=0.01)
print("Is head frozen in SCL before backward?", not list(model.head.parameters())[0].requires_grad)

