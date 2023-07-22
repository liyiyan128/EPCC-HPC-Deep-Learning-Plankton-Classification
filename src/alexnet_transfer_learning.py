import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import SGD
from torchvision import transforms, datasets, models

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.handlers import EarlyStopping, TerminateOnNan, ModelCheckpoint, Timer

import numpy as np
import os
from tqdm import tqdm


# Define train loop.
def train_loop(model_pkg, dataloaders, epochs=10, lr=0.001, momentum=0.9, log_interval=10):
    # Init the network.
    device = 'cpu'
    if torch.cuda.is_available():
        print('Train on cuda')
        device = 'cuda'

    # Push model into device.
    model = model_pkg[1]
    model = model.to(device)
    # Set Stochastic Gradient Descent parameters.
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)

    # Create a trainer engine.
    trainer = create_supervised_trainer(model, optimizer, F.cross_entropy, device=device)
    # Create an evaluator.
    evaluator = create_supervised_evaluator(model,
                                            metrics={'accuracy': Accuracy(),
                                                     'nll': Loss(F.cross_entropy)},
                                            device=device)

    # Format the progress bar
    # and print training metrics at completion of each epoch.
    itdesc = "ITERATION - loss: {:.2f}"
    pbar = tqdm(initial=0, leave=False,
                total=len(dataloaders['train']), desc=itdesc.format(0))

    # Attach a timer to time the training.
    timer = Timer()
    timer.attach(trainer)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(dataloaders['train']) + 1
        if iter % log_interval == 0:
            pbar.desc = itdesc.format(engine.state.output)
            pbar.update(log_interval)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        pbar.refresh()
        evaluator.run(dataloaders['train'])
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        train_avg_nll = metrics['nll']
        tqdm.write(
            "Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
            .format(engine.state.epoch, avg_accuracy*100, train_avg_nll)
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(dataloaders['valid'])
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        val_avg_nll = metrics['nll']
        tqdm.write(
            "Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.4f}"
            .format(engine.state.epoch, avg_accuracy*100, val_avg_nll)
        )
        pbar.n = pbar.last_print_n = 0

    # Save the model with the best log loss on the validation set.
    def score_function(engine):
        val_loss = evaluator.state.metrics['nll']
        return -val_loss

    modelhandler = ModelCheckpoint('transfer_learning_models', model_pkg[0],
                                   score_function=score_function, create_dir=True)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, modelhandler, {'mymodel': model})

    # Terminate training if NAN values are obtained.
    trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

    # Stop training if validation log loss does not improve.
    ES = EarlyStopping(patience=20, score_function=score_function, trainer=trainer)
    evaluator.add_event_handler(Events.COMPLETED, ES)

    # Run the training engine
    trainer.run(dataloaders['train'], max_epochs=epochs)

    # Print the time taken.
    print("Total time taken: ", timer.total, "seconds")

    pbar.close()


# Define data transforms and load data.
# Transform images to tensors and then normalise.
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
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
                for x in ['train', 'valid']}  # ,'test']}
dataloaders = {x: torch.utils.data.DataLoader(zooplankton[x], batch_size=24,
                                              shuffle=True, num_workers=4)
               for x in ['train', 'valid']}  # , 'test']}
print("Datasets loaded")

dataset_sizes = {x: len(zooplankton[x]) for x in ['train', 'valid']}  # , 'test']}
print(f"Sizes: {dataset_sizes}")
class_names = zooplankton['train'].classes
print(f"Classes: {class_names}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device available: {device}")


# Load AlexNet.
alexnet = models.alexnet()
print("Load AlexNet")
# Load pretrained weights.
alexnet.load_state_dict(torch.load("/work/m23ss/m23ss/liyiyan/EPCC-HPC-Deep-Learning-Plankton-Classification/src/torch/hub/checkpoints/alexnet-owt-7be5be79.pth"))
print("Pretrained weights loaded")

# Freeze model parameters.
for param in alexnet.parameters():
    param.requires_grad = False
# Replace the last fully-connected layer for transfer learning.
alexnet.classifier[-1] = nn.Linear(4096, 17)  # num_classes of zooplankton is 17
print("Refit AlexNet")

model = ["alexnet", alexnet]
print("Start train loop")
train_loop(model, dataloaders, epochs=100, lr=0.001, momentum=0.9)
print("Training completed")
