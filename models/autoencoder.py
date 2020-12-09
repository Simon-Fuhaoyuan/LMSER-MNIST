'''
Vanilla Auto Encoder
'''
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128, bias=False),
            nn.Sigmoid(),
            nn.Linear(128, 64, bias=False),
            nn.Sigmoid(),
            nn.Linear(64, 16, bias=False),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 64, bias=False),
            nn.Tanh(),
            nn.Linear(64, 128, bias=False),
            nn.Tanh(),
            nn.Linear(128, 28*28, bias=False),
            nn.Tanh(),
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x