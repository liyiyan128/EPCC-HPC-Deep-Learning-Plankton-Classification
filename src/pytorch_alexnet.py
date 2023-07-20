import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torchvision import transforms, datasets, models
from torchvision.utils import make_grid, save_image
# from random import randint

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.handlers import EarlyStopping, TerminateOnNan, ModelCheckpoint, Timer

import numpy as np
# import matplotlib.pyplot as plt
import os
# import PIL
from tqdm import tqdm


# Train loop.
def train_loop(model_pkg, dataloaders, epochs, lr=0.001, momentum=0.9, log_interval=10):
    # init the network:
    device = 'cpu'

    if torch.cuda.is_available():
        print('Train on cuda')
        device = 'cuda'

    # Push model into device.
    model = model_pkg[1]
    model = model.to(device)
    # set Stochastic Gradient Descent parameters:
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    # create a trainer engine:
    trainer = create_supervised_trainer(model, optimizer, F.cross_entropy, device=device)
    # and an evaluator:
    evaluator = create_supervised_evaluator(model,
                                            metrics={'accuracy': Accuracy(),
                                                     'nll': Loss(F.cross_entropy)},
                                            device=device)

    # The following is mostly formatting the progress bar (!)
    # and printing training metrics at completion of each epoch.
    itdesc = "ITERATION - loss: {:.2f}"
    pbar = tqdm(initial=0, leave=False, total=len(dataloaders['train']),
                desc=itdesc.format(0))

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
            .format(engine.state.epoch, avg_accuracy*100, val_avg_nll))
        pbar.n = pbar.last_print_n = 0

    # Save the model with the best log loss on the validation set.
    def score_function(engine):
        val_loss = evaluator.state.metrics['nll']
        return -val_loss

    modelhandler = ModelCheckpoint('transfer_learning_models', model_pkg[0], score_function=score_function, create_dir=True)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, modelhandler, {'mymodel': model})

    # Terminate training if NAN values are obtained.
    trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

    # Stop training if validation log loss does not improve.
    ES = EarlyStopping(patience=20, score_function=score_function, trainer=trainer)
    evaluator.add_event_handler(Events.COMPLETED, ES)

    # Run the training engine
    trainer.run(dataloaders['train'], max_epochs=epochs)

    # Print the time taken
    print("Total time taken: ", timer.total, "seconds")

    pbar.close()


class TransferModel(nn.Module):
    """This is a class for a transfer learning model.

    Creates a new NN model:
    keep the structure and the weights of the features layers
    and replace the original model classifier.
    """
    def __init__(self, original_model, classifier):
        super(TransferModel, self).__init__()

        self.features = original_model.features
        self.classifier = classifier

        # Freeze original weights.
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        feats = self.features(x)
        # Flatten network.
        feats = feats.view(feats.size(0), np.prod(feats.shape[1:]))
        y = self.classifier(feats)
        return y


# Transform images to tensors and then normalise.
# Input normalised as mini-batches of 3-chanel RGB images of shape (3 x H x W),
# where H and W are expected to be at least 224.
# Images have to be loaded into a range of [0, 1] and
# normalised using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
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
print("Dataset loaded.")
dataset_sizes = {x: len(zoo_datasets[x]) for x in ['train', 'valid']}  # , 'test']}
print(f"Dataset sizes: {dataset_sizes}")
class_names = zoo_datasets['train'].classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device available: {device}")

# out_features=17

# AlexNet.
alexnet = models.alexnet(pretrained=True)

classifier = nn.Sequential(
    nn.Dropout(),
    nn.Linear(256 * 6 * 6, 4096),
    nn.ReLU(inplace=True),
    nn.Dropout(),
    nn.Linear(4096, 4096),
    nn.ReLU(inplace=True),
    nn.Linear(4096, 17),
)

alexnet_transfer = TransferModel(alexnet, classifier)
model = ["alexnet_transfer", alexnet_transfer]
train_loop(alexnet_transfer, dataloaders, epochs=10, lr=0.001, momentum=0.9)
