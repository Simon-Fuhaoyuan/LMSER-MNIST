'''
LMSER with DCW nature, and with pseudo inverse constraint.
'''
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

class Lmser(nn.Module):
    def __init__(self):
        super(Lmser, self).__init__()
        self.l1 = nn.Linear(28*28, 128, bias=False)
        self.l2 = nn.Linear(128, 64, bias=False)
        self.l3 = nn.Linear(64, 16, bias=False)
        self.wlist = [self.l1.weight, self.l2.weight, self.l3.weight]
        self.dew = [None]*3

    @property
    def __name__(self):
        return 'Lmser'

    def forward(self, x):
        for i in range(3):
            (u, s, v) = torch.svd(self.wlist[i])
            mid = torch.eye(min(u.size())).cuda()
            self.dew[i]=v.mm(mid * (1 / s.unsqueeze(1))).mm(u.T)
        
        x = F.leaky_relu(self.l1(x))
        
        x = F.leaky_relu(self.l2(x))

        x = self.l3(x)

        x = F.leaky_relu(F.linear(x, self.dew[2]))

        x = F.leaky_relu(F.linear(x, self.dew[1]))

        x = F.linear(x, self.dew[0])
        
        return x

class LmserSupervise(nn.Module):
    def __init__(self):
        super(LmserSupervise, self).__init__()
        self.l1 = nn.Linear(28*28, 128, bias=False)
        self.l2 = nn.Linear(128, 64, bias=False)
        self.l3 = nn.Linear(64, 16, bias=False)
        self.classifier = nn.Linear(16, 10, bias=False)
        self.wlist = [self.l1.weight, self.l2.weight, self.l3.weight]
        self.dew = [None]*3

    @property
    def __name__(self):
        return 'LmserSupervise'

    def forward(self, x):
        for i in range(3):
            (u, s, v) = torch.svd(self.wlist[i])
            mid = torch.eye(min(u.size())).cuda()
            self.dew[i]=v.mm(mid * (1 / s.unsqueeze(1))).mm(u.T)
        
        x = F.leaky_relu(self.l1(x))
        
        x = F.leaky_relu(self.l2(x))

        x = self.l3(x)

        vec = self.classifier(x)

        x = F.leaky_relu(F.linear(x, self.dew[2]))

        x = F.leaky_relu(F.linear(x, self.dew[1]))

        x = F.linear(x, self.dew[0])
        
        return x, vec
