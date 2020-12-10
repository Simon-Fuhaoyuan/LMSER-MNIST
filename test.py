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

def l2_error(prediction, gt):
    prediction = prediction.cpu().detach().numpy()
    gt = gt.cpu().detach().numpy()
    assert prediction.shape == gt.shape, \
        'The prediction shape is {}, but gt shape is {}'.format(prediction.shape, gt.shape)

    error_mat = (prediction - gt) ** 2
    error = error_mat.mean()

    return error

def test(model, device, test_loader):
    model.eval()
    total_error = 0.0
    for i, (images, labels) in enumerate(test_loader):
        images = images.view(-1, 784).to(device)
        out = model(images)

        error = l2_error(out, images)
        total_error += error
    avg_error = total_error / len(test_loader)
    print('Test error is %.4f' % avg_error)
    
    return avg_error

if __name__ == '__main__':
    mean = 0.5
    std = 0.5
    epochs = 30
    batch_size = 64
    lr = 0.003

    transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean=(mean,),std=(std,))])
    test_dataset = datasets.MNIST(root='./data/', train=False, transform=transform, download=True)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    model = AutoEncoder().to(device) # vanilla AE
    # model = DCW_woConstraint()

    test(model, device, test_loader)
