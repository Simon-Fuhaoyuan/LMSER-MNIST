import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import torch.utils.data as data
import torch.nn.functional as F
import matplotlib.pyplot as plt

epochs = 30
batch_size = 64
lr = 0.003

def visulization(images):
    img = torchvision.utils.make_grid(images)
    img = img.numpy().transpose(1, 2, 0)
    std = 0.5
    mean = 0.5
    img = img * std + mean
    print(labels)
    plt.imshow(img)
    plt.show()

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
class DCW_woConstraint(nn.Module):
    def __init__(self):
        super(DCW_woConstraint, self).__init__()
        self.l1 = nn.Linear(28*28, 128, bias=False)
        self.l2 = nn.Linear(128, 64, bias=False)
        self.l3 = nn.Linear(64, 16, bias=False)
    
    def forward(self, x):
        w3 = self.l3.weight.T
        w2 = self.l2.weight.T
        w1 = self.l1.weight.T
        x = F.sigmoid(self.l1(x))
        # x = self.l1(x)
        x = F.sigmoid(self.l2(x))
        # x = self.l2(x)
        x = self.l3(x)
        x = F.tanh(F.linear(x, w3))
        # x = F.linear(x, w3)
        x = F.tanh(F.linear(x, w2))
        # x = F.linear(x, w2)
        x = F.tanh(F.linear(x, w1))
        # x = F.linear(x, w1)
        return x

if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean=(0.5,),std=(0.5,))])
    train_dataset = datasets.MNIST(root='./data/', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data/', train=False, transform=transform, download=True)
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    model = AutoEncoder() # vanilla AE
    # model = DCW_woConstraint()
    optim = torch.optim.Adam(model.parameters(), lr = lr)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.view(-1, 784)
            out = model(images)

            optim.zero_grad()
            loss = criterion(out, images)
            loss.backward()
            optim.step()
        print('[{}/{}] Loss:'.format(epoch+1, epochs), loss.item())
    images, labels = next(iter(test_loader))
    images = images.view(-1, 784)
    model.eval()
    out = model(images).view(batch_size, 1, 28, 28)
    visulization(out.detach())
