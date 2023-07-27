"""Evaluate AlexNet model.

The evaluation metrics used are accuracy, cross-entropy loss,
top-5 category accuracy and confusion matrix.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torchvision import transforms, datasets, models
from torchvision.utils import make_grid, save_image

from ignite.engine import Events, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, TopKCategoricalAccuracy, ConfusionMatrix
from ignite.handlers import Timer

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
from PIL import Image
from tqdm import tqdm


def model_evaluator(model, weights, dataset, top_k_class=5,
                    save_confusion=False,
                    saved_file_name="confusion_matrix.png"):
    """Evaluate the model.

    This function assumes that the dataset is loaded onto GPU.

    Parameters
    ----------
    model
        The model to be evaluated.
    weights
        The path to the weights file of the model.
    dataset
        The dataset on which the model is trained.
    top_k_class, default=5
        The top k classes in the dataset.
    """
    model.load_state_dict(torch.load(weights))
    model.eval()
    evaluator = create_supervised_evaluator(
        model,
        metrics={'accuracy': Accuracy(),
                 'nll': Loss(F.cross_entropy),
                 'top_k_accuracy': TopKCategoricalAccuracy(k=top_k_class),
                 'confusion_matrix': ConfusionMatrix(len(class_names))},
        device=device
    )
    timer = Timer()
    timer.attach(evaluator)

    evaluator.run(dataloaders[str(dataset)])
    metrics = evaluator.state.metrics
    avg_accuracy = metrics['accuracy']
    val_avg_nll = metrics['nll']
    confusion_matrix = metrics['confusionmatrix']
    top_k_accuracy = metrics['top_k_accuracy']
    confusion_numpy = confusion_matrix.numpy()

    print(
            "Model: {}".format(model.__name__)
         )
    print(
            "Total Time taken: {:.1f}".format(timer.total) + " seconds \n"
         )
    print(
            "Validation Results -  \n\n Avg Accuracy: {:.2f} \n Avg Loss: {:.4f} \n"
            .format(avg_accuracy*100, val_avg_nll)
         )
    print(
            "Avg Top " + str(top_k_class) + " Accuracy: {:.2f} \n"
            .format(top_k_accuracy*100)
         )

    mpl.style.use('seaborn')

    conf_arr = confusion_numpy

    df_cm = pd.DataFrame(conf_arr,
                         index=list(class_names),
                         columns=list(class_names)
                         )
    fig = plt.figure()

    plt.clf()

    ax = fig.add_subplot(111)
    ax.set_aspect(1)

    cmap = sns.cubehelix_palette(light=1, as_cmap=True)

    res = sns.heatmap(df_cm, annot=True, vmin=0.0, vmax=100.0, fmt='.0f', cmap=cmap)

    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    if save_confusion:
        plt.savefig(str(saved_file_name), dpi=100, bbox_inches='tight')


# Define data transforms and load data.
# Transform images to tensors and then normalise.
data_transforms = {
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

data_loc = '/work/m23ss/m23ss/liyiyan/plankton/CPRNet/data/'  # path to data

zooplankton = {x: datasets.ImageFolder(os.path.join(data_loc, x), data_transforms[x])
               for x in ['valid', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(zooplankton[x], batch_size=24,
                                              shuffle=True, num_workers=4)
               for x in ['valid', 'test']}
print("Datasets loaded")

dataset_sizes = {x: len(zooplankton[x]) for x in ['valid', 'test']}
print(f"Sizes: {dataset_sizes}")
class_names = zooplankton['valid'].classes
print(f"Classes: {class_names}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device available: {device}")


# Load AlexNet.
alexnet = models.alexnet()
print("Load AlexNet")
# # Load pretrained weights.
# alexnet.load_state_dict(torch.load("/work/m23ss/m23ss/liyiyan/EPCC-HPC-Deep-Learning-Plankton-Classification/src/torch/hub/checkpoints/alexnet-owt-7be5be79.pth"))
# print("Pretrained weights loaded")

# Freeze model parameters.
for param in alexnet.parameters():
    param.requires_grad = False
# Replace the last fully-connected layer for transfer learning.
alexnet.classifier[-1] = nn.Linear(4096, 17)  # num_classes of zooplankton is 17
# print("Refit AlexNet")


# Evaluate AlexNet
weights = "/work/m23ss/m23ss/liyiyan/EPCC-HPC-Deep-Learning-Plankton-Classification/src/torch/hub/checkpoints/alexnet-owt-7be5be79.pth"
dataset = "test"
model_evaluator(alexnet, weights, dataset, top_k_class=5,
                save_confusion=False, saved_file_name="confusion_matrix.png")
