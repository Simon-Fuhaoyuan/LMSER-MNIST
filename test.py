import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import torch.utils.data as data
import numpy as np
import os
import argparse

from models import *
from utils import vis_image
from utils import make_dirs

def parser_args():
    parser = argparse.ArgumentParser(description='Train reconstruction task on mnist dataset.')

    # Compulsory parameters
    parser.add_argument('model', help='The name of your model', type=str)
    parser.add_argument('weight', help='The weight file of CNN.', type=str)
    parser.add_argument('id', type=str)

    # Alternative parameters
    parser.add_argument('--dataset', help='The name of the dataset', default='mnist', type=str)
    parser.add_argument('--batch_size', help='Batch size', default=64, type=int)
    parser.add_argument('--mean', help='Mean of input data', default=0, type=float)
    parser.add_argument('--std', help='Std of input data', default=1, type=float)
    parser.add_argument('--image_dir', help='The directory to store images', default='images', type=str)
    parser.add_argument('--cpu_only', help='If specified, training and testing will only use cpu', action='store_true', default=False)
    
    config = parser.parse_args()

    return config

def l2_error(prediction, gt):
    prediction = prediction.cpu().detach().numpy()
    gt = gt.cpu().detach().numpy()
    assert prediction.shape == gt.shape, \
        'The prediction shape is {}, but gt shape is {}'.format(prediction.shape, gt.shape)

    error_mat = (prediction - gt) ** 2
    error = error_mat.mean()

    return error

def vis_test(config, model, device, test_loader, vis_cnt=1):
    model.eval()
    for i, (images, labels) in enumerate(test_loader):
        if i == vis_cnt:
            break
        images = images.view(-1, 784).to(device)
        labels = labels.to(device)

        out = model(images).view(config.batch_size, 1, 28, 28)
        make_dirs(os.path.join(config.image_dir, config.id))
        vis_image(config, out, labels, os.path.join(config.image_dir, config.id, str(i + 1) + '.png'))

def vis_gt(config, model, device, test_loader, vis_cnt=1):
    model.eval()

    for i, (images, labels) in enumerate(test_loader):
        if i == vis_cnt:
            break
        images = images.view(config.batch_size, 1, 28, 28)
        make_dirs(os.path.join(config.image_dir, config.dataset))
        vis_image(config, images, labels, os.path.join(config.image_dir, config.dataset, '%d.png' % (i + 1)))

def vis_test_supervise(config, model, device, test_loader, vis_cnt=1):
    model.eval()
    for i, (images, labels) in enumerate(test_loader):
        if i == vis_cnt:
            break
        images = images.view(-1, 784).to(device)
        labels = labels.to(device)

        out, _ = model(images)
        out = out.view(config.batch_size, 1, 28, 28)
        make_dirs(os.path.join(config.image_dir, config.id))
        vis_image(config, out, labels, os.path.join(config.image_dir, config.id, str(i + 1) + '.png'))

def test(config, model, device, test_loader):
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

def test_supervise(config, model, device, test_loader):
    model.eval()
    total_error = 0.0
    for i, (images, labels) in enumerate(test_loader):
        images = images.view(-1, 784).to(device)
        out, _ = model(images)

        error = l2_error(out, images)
        total_error += error
    avg_error = total_error / len(test_loader)
    print('Test error is %.4f' % avg_error)
    
    return avg_error

if __name__ == '__main__':
    config = parser_args()
    # if config.model == 'AutoEncoder':
    #     config.mean = 0.5
    #     config.std = 0.5
    transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean=(config.mean,),std=(config.std,))])

    test_dataset = None
    test_loader = None
    if config.dataset == 'mnist':
        test_dataset = datasets.MNIST(root='./data/', train=False, transform=transform, download=True)
    else:
        test_dataset = datasets.FashionMNIST(root='./data/', train=False, transform=transform, download=True)
        
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
    model.load_state_dict(torch.load(config.weight), strict=True)

    # test(config, model, device, test_loader)
    vis_test(config, model, device, test_loader, 10)
    # vis_gt(config, model, device, test_loader, 10)
