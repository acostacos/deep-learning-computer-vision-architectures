import torch.nn.functional as F
from torch import nn

# 256 × 256 input
class AlexNet(nn.Module):
    """
    AlexNet model from ImageNet classification with deep convolutional neural networks (Krizhevsky et al.).
    Adapted from https://d2l.ai/chapter_convolutional-modern/alexnet.html
    """
    def __init__(self, in_channels=3, out_channels=1000, device='cpu'):
        super.__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=96, kernel_size=11, stride=4, device=device)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2) # Max Pool was found to perform better compared to Average Pool
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=4, device=device)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=4, device=device),
            nn.Conv2d(in_channels=384, out_channels=192, kernel_size=3, stride=4, device=device),
            nn.Conv2d(in_channels=192, out_channels=256, kernel_size=3, stride=4, device=device),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fc = nn.Sequential(
            nn.Linear(in_features=2304, out_features=4096, device=device), nn.ReLU(),
            nn.Linear(in_features=4096, out_features=4096, device=device), nn.ReLU(),
            nn.Linear(in_features=4096, out_features=out_channels, device=device),
        )
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Add downsampling
        # Normalize after each convolution / linear
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.fc(x)
        return x

    def loss(self, pred, label):
        pass