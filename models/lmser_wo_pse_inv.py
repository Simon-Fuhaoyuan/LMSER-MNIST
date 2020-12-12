'''
LMSER with DCW nature, but without pseudo inverse constraint.
'''
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

class LmserWoPseudoInverse(nn.Module):
    def __init__(self):
        super(LmserWoPseudoInverse, self).__init__()
        self.l1 = nn.Linear(28*28, 128, bias=False)
        self.l2 = nn.Linear(128, 64, bias=False)
        self.l3 = nn.Linear(64, 16, bias=False)
    
    @property
    def __name__(self):
        return 'LmserWoPseudoInverse'

    def forward(self, x):
        w3 = self.l3.weight.T
        w2 = self.l2.weight.T
        w1 = self.l1.weight.T

        # x = F.sigmoid(self.l1(x))
        x = F.leaky_relu(self.l1(x))

        # x = F.sigmoid(self.l2(x))
        x = F.leaky_relu(self.l2(x))

        x = self.l3(x)

        # x = F.tanh(F.linear(x, w3))
        x = F.leaky_relu(F.linear(x, w3))

        # x = F.tanh(F.linear(x, w2))
        x = F.leaky_relu(F.linear(x, w2))

        # x = F.tanh(F.linear(x, w1))
        x = F.linear(x, w1)

        return x


class LmserWoPseudoInverseSupervise(nn.Module):
    def __init__(self):
        super(LmserWoPseudoInverseSupervise, self).__init__()
        self.l1 = nn.Linear(28*28, 128, bias=False)
        self.l2 = nn.Linear(128, 64, bias=False)
        self.l3 = nn.Linear(64, 16, bias=False)
        self.classifier = nn.Linear(16, 10, bias=False)
    
    @property
    def __name__(self):
        return 'LmserWoPseudoInverseSupervise'

    def forward(self, x):
        w3 = self.l3.weight.T
        w2 = self.l2.weight.T
        w1 = self.l1.weight.T

        # x = F.sigmoid(self.l1(x))
        x = F.leaky_relu(self.l1(x))

        # x = F.sigmoid(self.l2(x))
        x = F.leaky_relu(self.l2(x))

        x = self.l3(x)

        vec = self.classifier(x)

        # x = F.tanh(F.linear(x, w3))
        x = F.leaky_relu(F.linear(x, w3))

        # x = F.tanh(F.linear(x, w2))
        x = F.leaky_relu(F.linear(x, w2))

        # x = F.tanh(F.linear(x, w1))
        x = F.linear(x, w1)

        return x, vec
