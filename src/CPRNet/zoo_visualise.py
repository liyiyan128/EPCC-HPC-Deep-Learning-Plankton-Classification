"""
Visualise some images.
"""
import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.optim import SGD
from torchvision import transforms, datasets
from torchvision.utils import make_grid

# from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
# from ignite.metrics import Accuracy, Loss, TopKCategoricalAccuracy
# from ignite.handlers import EarlyStopping, TerminateOnNan, ModelCheckpoint, Timer

import numpy as np
import matplotlib.pyplot as plt
import os
# from tqdm import tqdm


# Image data loaders.
# At minimum, images need to be transformed to tensors and then normalized.
data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

data_loc = '/work/m23ss/m23ss/liyiyan/plankton/CPRNet/data/'

zoo_datasets = {x: datasets.ImageFolder(os.path.join(data_loc, x), data_transforms[x])
                for x in ['train', 'valid']}  # ,'test']}
dataloaders = {x: torch.utils.data.DataLoader(zoo_datasets[x], batch_size=24,
                                              shuffle=True, num_workers=4)
               for x in ['train', 'valid']}  # , 'test']}
dataset_sizes = {x: len(zoo_datasets[x]) for x in ['train', 'valid']}  # , 'test']}
class_names = zoo_datasets['train'].classes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    # if file_name:
    #     plt.savefig(f"{file_name}.pdf")


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))
# to save figure space, only view first 5
first5 = inputs[:5, :, :, :]
classes_first5 = classes[:5]

# Make a grid from batch
out = make_grid(first5)

plt.figure(figsize=(40, 40))
imshow(out, title=[class_names[x] for x in classes_first5], file_name=[])
plt.savefig("zoo_visualisation_0.png")
