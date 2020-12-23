'''
Description: This file is to train a reconstruction network with a given model.
Usage: python train.py $MODEL_NAME [alternatie_parameters]
    Please make sure $MODEL_NAME is the same with that in models folder.
Parameters: Details in parser_args() function.
Behaviors: After training, this code will automatically make a folder at ./${weight_dir}/${current_time}/
    This folder will copy your config, and store the best model, as well as training loss, test error, and sample images.
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
from utils import save_best_model, make_dirs, save_config
from utils import vis_error, vis_loss, vis_loss_error, vis_image
from test import test, vis_test

def parser_args():
    parser = argparse.ArgumentParser(description='Train reconstruction task on mnist dataset.')

    # Compulsory parameters
    parser.add_argument('model', help='The name of your model', type=str)

    # Alternative parameters
    parser.add_argument('--epochs', help='Total epoches', default=50, type=int)
    parser.add_argument('--batch_size', help='Batch size', default=64, type=int)
    parser.add_argument('--lr', help='Learning Rate', default=0.003, type=float)
    parser.add_argument('--mean', help='Mean of input data', default=0, type=float)
    parser.add_argument('--std', help='Std of input data', default=1, type=float)
    parser.add_argument('--dataset', help='The name of the dataset', default='mnist', type=str)
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
            error = test(config, model, device, test_loader)
            errors.append(error)
            if error < min_error:
                min_error = error
                min_error_epoch = epoch + 1
                save_best_model(model, config.weight_dir)
                vis_test(config, model, device, test_loader)
    
    return losses, errors, min_error, min_error_epoch

if __name__ == '__main__':
    config = parser_args()

    # get current time
    time_str = time.strftime("%m-%d-%H-%M-%S", time.localtime())
    config.weight_dir = os.path.join(config.weight_dir, time_str)
    make_dirs(config.weight_dir)
    save_config(config)

    # transform
    transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean=(config.mean,),std=(config.std,))])
    
    # train, test loader
    train_dataset = None
    test_dataset = None
    train_loader = None
    test_loader = None
    if config.dataset == 'mnist':
        train_dataset = datasets.MNIST(root='./data/', train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST(root='./data/', train=False, transform=transform, download=True)
    else:
        train_dataset = datasets.FashionMNIST(root='./data/', train=True, transform=transform, download=True)
        test_dataset = datasets.FashionMNIST(root='./data/', train=False, transform=transform, download=True)
        
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)
    if not config.no_test:
            test_loader = data.DataLoader(dataset=test_dataset, batch_size=config.batch_size, shuffle=False)

    # device, using cpu or cuda
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

    # model, optimizer and loss
    model = eval(config.model)().to(device)
    optim = torch.optim.Adam(model.parameters(), lr = config.lr)
    # criterion = nn.MSELoss().to(device)
    criterion = nn.L1Loss(reduction='mean').to(device)

    losses, errors, min_error, min_error_epoch = train(config, model, optim, criterion, device, train_loader, test_loader)
    vis_loss(config, losses)
    if not config.no_test:
        vis_error(config, errors)
        vis_loss_error(config, losses, errors)

    print('Min error: %.4f, min error epoch: %d' % (min_error, min_error_epoch))
