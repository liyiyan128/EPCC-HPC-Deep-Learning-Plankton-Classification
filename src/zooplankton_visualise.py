"""
Visualise some images in the Zooplankton dataset.
"""
import torch
from torchvision import transforms, datasets
from torchvision.utils import make_grid

import numpy as np
import matplotlib.pyplot as plt
import os


# Transform images to tensors and then normalise.
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

data = {x: datasets.ImageFolder(os.path.join(data_loc, x), data_transforms[x])
                for x in ['train', 'valid']}  # ,'test']}

dataloaders = {x: torch.utils.data.DataLoader(data[x], batch_size=24,
                                              shuffle=True, num_workers=4)
               for x in ['train', 'valid']}  # , 'test']}
print("Datasets loaded")

dataset_sizes = {x: len(data[x]) for x in ['train', 'valid']}  # , 'test']}
print(f"Dataset sizes: {dataset_sizes}")
class_names = data['train'].classes
print(f"Classes: {class_names}")


def imshow(inp, title=None):
    """imshow for tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data.
inputs, classes = next(iter(dataloaders['train']))
# To save figure space, only view first 5.
first5 = inputs[:5, :, :, :]
classes_first5 = classes[:5]

# Make a grid from batch.
out = make_grid(first5)
# Plot.
plt.figure(figsize=(40, 40))
imshow(out, title=[class_names[x] for x in classes_first5])
print("Plots completed")
plt.savefig("/work/m23ss/m23ss/liyiyan/EPCC-HPC-Deep-Learning-Plankton-Classification/src/planktons.pdf")
print("Figures saved")
