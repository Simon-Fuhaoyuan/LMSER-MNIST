import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('AGG')
import numpy as np
import torchvision

def vis_image(config, prediction, labels, name=None):
    images = prediction.cpu().detach()
    img = torchvision.utils.make_grid(images)
    img = img.numpy().transpose(1, 2, 0)
    img = img * config.std + config.mean
    img = (img - img.min()) / (img.max() - img.min())
    if name is None:
        matplotlib.image.imsave(os.path.join(config.weight_dir, 'result.png'), img)
    else:
        matplotlib.image.imsave(name, img)

def vis_loss_error(config, losses, errors):
    losses = np.array(losses)
    errors = np.array(errors)
    epochs = np.linspace(1, config.epochs, config.epochs)
    plt.plot(epochs, losses, label='Training loss')
    plt.plot(epochs, errors, label='Testing error')
    plt.legend()
    plt.title('Loss and error for %s' % config.model)
    plt.xlabel('Epochs')
    plt.ylabel('Loss/Error')
    plt.savefig(os.path.join(config.weight_dir, 'loss_error.png'))
    plt.close()

def vis_loss(config, losses, title=None, ylabel=None, name=None):
    losses = np.array(losses)
    epochs = np.linspace(1, config.epochs, config.epochs)
    plt.plot(epochs, losses)
    plt.legend()

    if title is None:
        plt.title('Loss for training %s' % config.model)
    else:
        plt.title(title)
    
    plt.xlabel('Epochs')
    plt.ylabel(ylabel if ylabel is not None else 'Loss')

    fig_name = ''
    if name is None:
        fig_name = os.path.join(config.weight_dir, 'loss.png')
    else:
        fig_name = os.path.join(config.weight_dir, name + '.png')
    plt.savefig(fig_name)
    plt.close()

def vis_error(config, errors):
    vis_loss(
        config, 
        errors, 
        title='Error for testing %s' % config.model,
        ylabel='Error',
        name='error'
    )
