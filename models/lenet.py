import torch.nn.functional as F
from torch import nn

class LeNet(nn.Module):
    """ Lenet-5 model. Taken from https://d2l.ai/chapter_convolutional-neural-networks/lenet.html """
    def __init__(self, in_shape, out_shape, lr=0.1, device='cpu'):
        super().__init__()

        self.in_shape = in_shape
        self.out_shape = out_shape
        self.lr = lr

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2, device=device)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, device=device)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc = nn.Sequential(
            nn.Linear(in_features=16*5*5, out_features=120, device=device), nn.Sigmoid(),
            nn.Linear(in_features=120, out_features=84, device=device), nn.Sigmoid(),
            nn.Linear(in_features=84, out_features=self.out_shape, device=device),
        )
    
    def forward(self, x):
        x = F.sigmoid(self.conv1(x))
        x = self.pool1(x)
        x = F.sigmoid(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 16*5*5)
        x = self.fc(x)
        return x
    
    def loss(self, pred, label):
        return F.cross_entropy(pred, label)
