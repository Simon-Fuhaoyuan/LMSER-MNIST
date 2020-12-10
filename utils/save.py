import os
import torch

def make_dirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def save_model(model, dir_name, suffix=None):
    make_dirs(dir_name)
    if suffix is not None:
        if '.pth' in suffix:
            path = os.path.join(dir_name, model.__name__ + suffix)
        else:
            path = os.path.join(dir_name, model.__name__ + suffix + '.pth')
    else:
        path = os.path.join(dir_name, model.__name__ + '.pth')
    
    print('Saving model at path %s.' % path)
    torch.save(model.state_dict(), path)

def save_best_model(model, dir_name):
    save_model(model, dir_name, '_best')
