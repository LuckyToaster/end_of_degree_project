from torch import nn
from torch import flatten

class Regressor(nn.Module):
    def __init__(self, input_dim, n_targets, dropout):
        super(Regressor, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1) 
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(input_dim//8 * input_dim//8, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, n_targets)
        self.drop = nn.Dropout(dropout)
        self.leaky = nn.LeakyReLU()

    def forward(self, x):
        x = self.maxpool(self.bn1(self.leaky(self.conv1(x))))
        x = self.maxpool(self.bn2(self.leaky(self.conv2(x))))
        x = self.maxpool(self.bn3(self.leaky(self.conv3(x))))
        x = flatten(x, 1)
        x = self.leaky(self.fc1(x))
        x = self.drop(self.leaky(self.fc2(x)))
        x = self.fc3(x)
        return x
