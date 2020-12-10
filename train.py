import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import torch.utils.data as data
import numpy as np
import os

from models import AutoEncoder
from models import LmserWoPseudoInverse
from utils import save_best_model
from test import test

def train(model, epochs, optim, criterion, device, train_loader, test_loader=None):
    save_dir = 'save_dir'
    losses = []
    model.train()
    min_loss = 999
    min_loss_epoch = 0
    total_loss = 0.0

    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.view(-1, 784).to(device)
            out = model(images)

            optim.zero_grad()
            loss = criterion(out, images)
            loss.backward()
            optim.step()

            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader) # avg loss in an image
        print('[{}/{}] Loss:'.format(epoch + 1, epochs), avg_loss)
        losses.append(avg_loss)
        total_loss = 0.0

        if avg_loss < min_loss:
            min_loss = avg_loss
            min_loss_epoch = epoch + 1
            save_best_model(model, save_dir)
    
    return losses

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    model = AutoEncoder().to(device) # vanilla AE
    # model = DCW_woConstraint()
    optim = torch.optim.Adam(model.parameters(), lr = lr)
    criterion = nn.MSELoss().to(device)

    train(model, epochs, optim, criterion, device, train_loader, test_loader)
