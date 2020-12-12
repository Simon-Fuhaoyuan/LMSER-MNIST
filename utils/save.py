import os
import torch
import numpy as np

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

def save_config(config):
    config_dict = config.__dict__
    config_file = os.path.join(config.weight_dir, 'config')
    with open(config_file, 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in config_dict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')

def save_array(array, name):
    array = np.array(array)
    np.save(name, array)
