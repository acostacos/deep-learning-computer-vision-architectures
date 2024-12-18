import torch.nn.functional as F
from torch import nn

class LeNet(nn.Module):
    """
    Lenet-5 model from Gradient-based learning applied to document recognition (LeCun et al.).
    Adapted from https://d2l.ai/chapter_convolutional-neural-networks/lenet.html
    """
    def __init__(self, in_channels=1, out_channels=10, device='cpu'):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5, padding=2, device=device)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, device=device)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc = nn.Sequential(
            nn.Linear(in_features=16*5*5, out_features=120, device=device), nn.Sigmoid(),
            nn.Linear(in_features=120, out_features=84, device=device), nn.Sigmoid(),
            nn.Linear(in_features=84, out_features=out_channels, device=device),
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.sigmoid(self.conv1(x))
        x = self.pool1(x)
        x = self.sigmoid(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 16*5*5)
        x = self.fc(x)
        return x
    
    def loss(self, pred, label):
        return F.cross_entropy(pred, label)
