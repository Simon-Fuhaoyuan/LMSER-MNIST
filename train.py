import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import torch.utils.data as data
import numpy as np

from models import AutoEncoder
from models import LmserWoPseudoInverse
from utils import *

def train(model, epochs, optim, criterion, train_loader, test_loader=None):
    min_loss = 999
    min_loss_epoch = 0
    
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.view(-1, 784)
            out = model(images)

            optim.zero_grad()
            loss = criterion(out, images)
            loss.backward()
            optim.step()
        
        print('[{}/{}] Loss:'.format(epoch + 1, epochs), loss.item())

if __name__ == '__main__':
    mean = 0.5
    std = 0.5
    epochs = 30
    batch_size = 64
    lr = 0.003

    transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean=(mean,),std=(std,))])
    train_dataset = datasets.MNIST(root='./data/', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data/', train=False, transform=transform, download=True)
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    model = AutoEncoder() # vanilla AE
    print(model.__name__)
    # model = DCW_woConstraint()
    optim = torch.optim.Adam(model.parameters(), lr = lr)
    criterion = nn.MSELoss()

    train(model, epochs, optim, criterion, train_loader, test_loader)
