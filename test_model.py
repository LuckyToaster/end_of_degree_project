import torch
from src.helpers.models import get_EfficientNet_B3

model, transforms = get_EfficientNet_B3()
x = torch.randn(2, 3, 224, 224)
out = model(x)
print("Output shape:", out.shape)
print("Output:", out)
