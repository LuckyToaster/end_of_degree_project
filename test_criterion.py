import torch
import torch.nn as nn

class CustomCriterion(torch.nn.Module):
    def __init__(self, criterion_function, alpha=0.1, beta=0.5):
        super().__init__()
        self.criterion = criterion_function
        self.alpha = alpha
        self.beta = beta

    def forward(self, preds, targets): 
        mse_loss = self.criterion(preds, targets)
        pred_f = preds[:, 0]
        pred_p = preds[:, 1]
        pred_c = preds[:, 2]
        pred_cal = preds[:, 3]
        supposed_cal = (4 * pred_p) + (4 * pred_c) + (9 * pred_f) 
        constraint_loss = self.criterion(supposed_cal, pred_cal)
        return (self.alpha * mse_loss) + (self.beta * constraint_loss)

criterion = CustomCriterion(nn.HuberLoss())
preds = torch.randn(1, 4)
# NO requires_grad on preds!
targets = torch.randn(1, 4)
loss = criterion(preds, targets)
try:
    loss.backward()
    print("BACKWARD SUCCESS")
except Exception as e:
    print(f"BACKWARD FAILED: {e}")
