# Deep Learning: Image Classification Using CNN

This is my EPCC (University of Edinburgh) HPC summer school project on Deep Learning: Image Classification Using CNN on the Zooplankton Dataset.
This project is based on the work of Tong (MSc in High Performance Computing at EPCC). [[1]](#reference) The code for `train_loop` and `model_evaluator` is from Tong's work.
This work used the Cirrus UK National Tier-2 HPC Service at EPCC (http://www.cirrus.ac.uk) funded by the University of Edinburgh and EPSRC (EP/P020267/1).

The model training and other computationally intensive work were done on Cirrus. Therefore, I used Python scripts instead of Jupyter notebooks, which made it easier for me to work with the job scheduler (Slurm).

In this project, AlexNet was used as a Transfer Learning model and was compared to VGG.

Contents:

- [Deep Learning: Image Classification Using CNN](#deep-learning-image-classification-using-cnn)
  - [AlexNet](#alexnet)
  - [Transfer Learning](#transfer-learning)
    - [Implement AlexNet as a Transfer Learning Model](#implement-alexnet-as-a-transfer-learning-model)
  - [Zooplankton Dataset](#zooplankton-dataset)
  - [Reference](#reference)

## AlexNet

*AlexNet* is a deep convolutional neural network (CNN) developed by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton in 2012. [[2]](#reference)

AlexNet was one of the first Deep Learning models that utilised a deep architecture with multiple layers, which allows complex hierarchical features to be captured. AlexNet architecture consists of eight layers: five convolutional layers and three fully-connected layers. Each convolutional layer applies a set of learnable filters (kernels) to the input image. These filters detect various patterns and features in different regions of the image.

Below list some key features introduced by AlexNet.

- **ReLU activation:** AlexNet uses ReLU (Rectified Linear Units) to mitigate the vanishing gradient problem. In addition, ReLu accelerates the training process by introducing non-linearity in the network.
- **Overlapping pooling:** models with overlapping pooling are generally harder to overfit.
- **GPU utilisation:** AlexNet allows for multi-GPU training by putting half of the model's neurons on GPU and the other half on another GPU. AlexNet was one of the first deep learning models to fully harness the power of GPU.

## Transfer Learning

Training a deep neural network from scratch on a specific task usually requires large amounts of labeled data and computational resources. *Transfer Learning* (TL) is a ground-breaking machine learning technique that leverages knowledge gained from pre-trained models and adapts it to new tasks with limited data, ultimately improving both efficiency and performance.

### Implement AlexNet as a Transfer Learning Model

More details can be found in [alexnet_transfer_learning.py](https://github.com/liyiyan128/EPCC-HPC-Deep-Learning-Plankton-Classification/blob/main/src/alexnet_transfer_learning.py).

## Zooplankton Dataset

- The images are of size $255 \times 255$ pixels (standard input size for most CNN architectures).
- There are 43 classes of zooplankton. Some classes have as little as 15 images, which is insufficient for machine learning tasks. Therefore, the dataset is reduced to only 17 classes with at least 1000 images for each class.

Below is a visualisation of the Zooplankton dataset: Images of Zooplanktons from Five Classes (cladocera, centropages, cladocera, jellyfish_type1, shrimp)
![planktons](https://github.com/liyiyan128/EPCC-HPC-Deep-Learning-Plankton-Classification/blob/main/src/planktons.png "Images of Zooplanktons from Five Classes.")

## Reference

[1] Tong Y Y. Investigating Deep Learning Approaches for Robust Zooplankton Identification [master's thesis]. The University of Edinburgh; 2019.

[2] Krizhevsky A, Sutskever I, Hinton GE. ImageNet Classification with Deep Convolutional Neural Networks. In: Pereira F, Burges CJC, Bottou L, Weinberger KQ, editors. Advances in Neural Information Processing Systems 25 (NIPS 2012). Curran Associates, Inc.; 2012. p. 1097–1105.
