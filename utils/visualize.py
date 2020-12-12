import matplotlib.pyplot as plt
import numpy as np
import torchvision

def visulization(prediction, labels, mean=0.5, std=0.5):
    images = prediction.cpu().detach()
    img = torchvision.utils.make_grid(images)
    img = img.numpy().transpose(1, 2, 0)
    img = img * std + mean
    print(labels)
    plt.imshow(img)
    plt.show()