'''
Description: This file is to train a reconstruction network with a given model.
Usage: python train.py $MODEL_NAME [alternatie_parameters]
    Please make sure $MODEL_NAME is the same with that in models folder.
Parameters: Details in parser_args() function.
Behaviors: After training, this code will automatically make a folder at ./${weight_dir}/${current_time}/
    This folder will copy your model name, config, and store the best model.
'''

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import torch.utils.data as data
import numpy as np
import os
import argparse
import time

from models import *
from utils import save_best_model, make_dirs
from test import test

def parser_args():
    parser = argparse.ArgumentParser(description='Train reconstruction task on mnist dataset.')

    # Compulsory parameters
    parser.add_argument('model', help='The name of your model', type=str)

    # Alternative parameters
    parser.add_argument('--epochs', help='Total epoches', default=30, type=int)
    parser.add_argument('--batch_size', help='Batch size', default=64, type=int)
    parser.add_argument('--lr', help='Learning Rate', default=0.003, type=float)
    parser.add_argument('--mean', help='Mean of input data', default=0.5, type=float)
    parser.add_argument('--std', help='Std of input data', default=0.5, type=float)
    parser.add_argument('--weight_dir', help='The directory to store CNN parameters', default='save_dir', type=str)
    parser.add_argument('--image_dir', help='The directory to store images', default='images', type=str)
    parser.add_argument('--cpu_only', help='If specified, training and testing will only use cpu', action='store_true', default=False)
    parser.add_argument('--no_test', help='If specified, there will be no test during training', action='store_true', default=False)
    
    config = parser.parse_args()

    return config

def train(config, model, optim, criterion, device, train_loader, test_loader=None):
    losses = []
    errors = []
    model.train()
    min_loss = 999
    min_loss_epoch = 0
    total_loss = 0.0

    min_error = 999
    min_error_epoch = 0

    for epoch in range(config.epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.view(-1, 784).to(device)
            out = model(images)

            optim.zero_grad()
            loss = criterion(out, images)
            loss.backward()
            optim.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader) # avg loss in an image
        print('[{}/{}] Loss:'.format(epoch + 1, config.epochs), avg_loss)
        losses.append(avg_loss)
        total_loss = 0.0

        if config.no_test or test_loader is None:
            if not config.no_test:
                print('Warning: you set test during training by default, but not specify a test loader.')
                print('If you want no test, please specify --no_test.')
            if avg_loss < min_loss:
                min_loss = avg_loss
                min_loss_epoch = epoch + 1
                save_best_model(model, config.weight_dir)
        else:
            error = test(model, device, test_loader)
            errors.append(error)
            if error < min_error:
                min_error = error
                min_error_epoch = epoch + 1
                save_best_model(model, config.weight_dir)
    
    return losses, errors

if __name__ == '__main__':
    config = parser_args()

    # get current time
    time_str = time.strftime("%m-%d-%H-%M-%S", time.localtime())
    config.weight_dir = os.path.join(config.weight_dir, time_str)
    make_dirs(config.weight_dir)
    # print(config)

    transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean=(config.mean,),std=(config.std,))])
    train_dataset = datasets.MNIST(root='./data/', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data/', train=False, transform=transform, download=True)
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=config.batch_size, shuffle=False)

    device = None
    if config.cpu_only:
        device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            print('Warning: you use CUDA by default, but CUDA is not available. This code will use CPU.')
            print('If you want to use cpu only, please specify --cpu_only.')
            device = torch.device('cpu')

    model = eval(config.model)().to(device)

    optim = torch.optim.Adam(model.parameters(), lr = config.lr)
    criterion = nn.MSELoss().to(device)

    losses, errors = train(config, model, optim, criterion, device, train_loader, test_loader)
