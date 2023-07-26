# Deep Learning: Image Classification Using CNN

This is my EPCC (University of Edinburgh) HPC summer school project on Deep Learning: Image Classification Using CNN on the Zooplankton Dataset. This work used the Cirrus UK National Tier-2 HPC Service at EPCC (http://www.cirrus.ac.uk) funded by the University of Edinburgh and EPSRC (EP/P020267/1).

The model training and other computational intensive work was done on Cirrus. Therefore, I used Python scripts instead of Jupyter notebooks, which made it easier for me to work with the job scheduler (Slurm).

In this project, AlexNet was used as a Transfer Learning model.

Contents:

- [Deep Learning: Image Classification Using CNN](#deep-learning-image-classification-using-cnn)
  - [AlexNet](#alexnet)
  - [Zooplankton Dataset](#zooplankton-dataset)
  - [Reference](#reference)

## AlexNet

AlexNet is a deep convolutional neural network (CNN) developed by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton in 2012. [[1]](#reference)

AlexNet was one of the first Deep Learning models that utilised a deep architecture with multiple layers, which allows complex hierachical features to be captured. AlexNet architecture consists of eight layers: five convolutional layers and three fully-connected layers. Each convolutional layer applies a set of learnable filters (kernels) to the input image. These filters detect various patterns and features in different regions of the image.

Below list some key features introduced by AlexNet.

- **ReLU activation:** AlexNet uses ReLU (Rectified Linear Units) to mitigate the vanishing gradient problem. In addition, ReLu accelerates the training process by introducing non-linearity in the network.
- **Overlapping pooling:** models with overlapping pooling are generally harder to overfit.
- **GPU utilisation:** AlexNet allows for multi-GPU traning by putting half of the model's neurons on GPU and the other half on another GPU. AlexNet was one of the first deep learning models to fully harness the power of GPU.

## Zooplankton Dataset

- The images are of size $255 \times 255$ pixels (standard input size for most CNN architextures).
- There are 43 classes of zooplankton. Some classes have as little as 15 images, which is insufficient for machine learning tasks. Therefore, the dataset is reduced to only 17 classes with at least 1000 images for each class.

![Visualisation of the Zooplankton dataset: five images from five different classes](https://github.com/liyiyan128/EPCC-HPC-Deep-Learning-Plankton-Classification/blob/main/src/planktons.pdf)

## Reference

[1] Krizhevsky A, Sutskever I, Hinton GE. ImageNet Classification with Deep Convolutional Neural Networks. In: Pereira F, Burges CJC, Bottou L, Weinberger KQ, editors. Advances in Neural Information Processing Systems 25 (NIPS 2012). Curran Associates, Inc.; 2012. p. 1097–1105.
