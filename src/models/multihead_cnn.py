from torch import nn
import torchvision


class MultiHeadCNN(nn.Module):
    def __init__(self, num_heads=3, num_classes=[10, 5, 2]):
        super().__init__()
        self.backbone = torchvision.models.resnet18(pretrained=True) # or resnet 50
        # Remove final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        # Define multiple heads
        self.heads = nn.ModuleList([
            nn.Linear(512, num_classes[i]) for i in range(num_heads)
        ])
    
    def forward(self, x):
        features = self.backbone(x).flatten(1)
        outputs = [head(features) for head in self.heads]
        return outputs   
