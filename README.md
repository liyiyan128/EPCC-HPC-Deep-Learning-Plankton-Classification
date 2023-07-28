# Deep Learning: Image Classification Using CNN

This is my EPCC (University of Edinburgh) HPC summer school project on Deep Learning: Image Classification Using CNN on the Zooplankton Dataset.

## Contents

- [Deep Learning: Image Classification Using CNN](#deep-learning-image-classification-using-cnn)
  - [Contents](#contents)
  - [Introduction](#introduction)
  - [Transfer Learning](#transfer-learning)
    - [AlexNet](#alexnet)
    - [VGG](#vgg)
    - [Implementing Transfer Learning](#implementing-transfer-learning)
  - [Model Performance](#model-performance)
    - [Confusion Matrix](#confusion-matrix)
  - [Zooplankton Dataset](#zooplankton-dataset)
  - [Reference](#reference)

## Introduction

This project is based on the work of Tong (MSc in High Performance Computing at EPCC). [[1]](#reference) The code for `train_loop` and `model_evaluator` is from Tong's work.
This work used the Cirrus UK National Tier-2 HPC Service at EPCC (http://www.cirrus.ac.uk) funded by the University of Edinburgh and EPSRC (EP/P020267/1).

The model training and other computationally intensive work were done on Cirrus. Therefore, I used Python scripts instead of Jupyter notebooks, which made it easier for me to work with the job scheduler (Slurm).

In this project, AlexNet was used as a Transfer Learning model and was compared to VGG. The two CNN architectures were trained to classify zooplankton from the Zooplankton dataset.

## Transfer Learning

Training a deep neural network from scratch on a specific task usually requires large amounts of labeled data and computational resources. *Transfer Learning* (TL) is a ground-breaking machine learning technique that leverages knowledge gained from pre-trained models and adapts it to new tasks with limited data, ultimately improving both efficiency and performance.

### AlexNet

*AlexNet* is a deep convolutional neural network (CNN) developed by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton in 2012. [[2]](#reference) The network was designed to participate in the ImageNet Large Scale Visual Recognition Challenge 2012 (ILSVR2012).

AlexNet was one of the first Deep Learning models that utilised a deep architecture with multiple layers, which allows complex hierarchical features to be captured. AlexNet architecture consists of eight layers: five convolutional layers and three fully-connected layers. Each convolutional layer applies a set of learnable filters (kernels) to the input image. These filters detect various patterns and features in different regions of the image.

Below lists some key features introduced by AlexNet.

- **ReLU activation**: AlexNet uses ReLU (Rectified Linear Units) to mitigate the vanishing gradient problem. In addition, ReLu accelerates the training process by introducing non-linearity in the network.
- **Overlapping pooling**: models with overlapping pooling are generally harder to overfit.
- **GPU utilisation**: AlexNet allows for multi-GPU training by putting half of the model's neurons on GPU and the other half on another GPU. AlexNet was one of the first deep learning models to fully harness the power of GPU.

### VGG

The VGG (Visual Geometry Group) was proposed by the Visual Geometry Group at the University of Oxford in 2014 to compete in ILSVRC. [[3]](#reference) The VGG architecture consists of a series of convolutional layers with $3 \times 3$ filters and max-pooling layers.

Below lists some key features of VGG.

- **Depth**: VGG demonstrated that deeper neural networks could lead to better performance.
- **Configurations**: VGG has various configurations with different numbers of weight layers. The deeper architecture generally produces slightly better performance at the cost of computational complexity.

VGG11 is used in this project, which has 11 layers.

### Implementing Transfer Learning

The typical process of transfer learning involves:

1. **Pre-training**: A large neural network is trained on a source task with a large amount of labeled data.
2. **Feature extraction**: The layers of the pre-trained neural network act as a feature extractor.
3. **Fine tuning**: In transfer learning, the weights of the pre-trained model are often retained. The final fully-connected layers are replaced and trained on the new dataset to adapt the model to the target task.

AlexNet and VGG were both trained on the ImageNet dataset. The last fully-connected layer is replaced by a new fully-connected layer with the number of outputs set to the number of classes in the Zooplankton dataset. The rest of the network is frozen and acts as a fixed feature extractor. The new classifier is then trained on the Zooplankton dataset.

More details can be found in [alexnet_transfer_learning.py](https://github.com/liyiyan128/EPCC-HPC-Deep-Learning-Plankton-Classification/blob/main/src/alexnet_transfer_learning.py).

## Model Performance

From empirical results, the model will converge quickly. Hence, the number of epochs is set to 100. The models are trained using fixed hyperparameters. The model evaluation results are shown in the table below.

| Model   | Validation Accuracy (%) | Cross-Entropy Loss | Epochs | Time Taken (s) |
| ---     | :---:                   | :---:              | :---:  | :---:          |
| AlexNet | 95.29                   | 0.1331             | 47     | 2803           |
| VGG11   | 97.47                   | 0.0813             | 91     | 6218           |

Although VGG produces better accuracy, it is notable that AlexNet only takes 47 epochs and hits early stopping conditions. It seems that AlexNet model converges faster on the Zooplankton dataset than VGG, leading to significantly shorter training time. The cost of shorter training time is about just 2% accuracy. However, there is no strict rule for model selection. Over 90% accuracy in machine learning is usualy considered very good. The biological aspect of this classification task may require high accuracy, while there are also economic concerns about computational cost.

### Confusion Matrix

A confusion matrix is a common machine learning model performance evaluation tool. It shows the number of correct and incorrect classifications made by the model for each class.

A visualisation of the confusion matrix for VGG11 is shown below.
![vgg11_confusion_matrix](https://github.com/liyiyan128/EPCC-HPC-Deep-Learning-Plankton-Classification/blob/main/src/vgg_confusion_matrix.png "Fig 1. VGG11 Confusion Matrix")

## Zooplankton Dataset

- The images are of size $255 \times 255$ pixels (standard input size for most CNN architectures).
- There are 43 classes of zooplankton. Some classes have as little as 15 images, which is insufficient for machine learning tasks. Therefore, the dataset is reduced to only 17 classes with at least 1000 images for each class. [[1]](#reference)

Below is a visualisation of the Zooplankton dataset: Images of Zooplanktons from Five Classes (cladocera, centropages, cladocera, jellyfish_type1, shrimp)
![planktons](https://github.com/liyiyan128/EPCC-HPC-Deep-Learning-Plankton-Classification/blob/main/src/planktons.png "Fig 2. Images of Zooplanktons from Five Classes.")

## Reference

[1] Tong Y Y. Investigating Deep Learning Approaches for Robust Zooplankton Identification [master's thesis]. The University of Edinburgh; 2019.

[2] Krizhevsky A, Sutskever I, Hinton GE. ImageNet Classification with Deep Convolutional Neural Networks. In: Pereira F, Burges CJC, Bottou L, Weinberger KQ, editors. Advances in Neural Information Processing Systems 25 (NIPS 2012). Curran Associates, Inc.; 2012. p. 1097–1105.

[3] Simonyan K, Zisserman A. Very Deep Convolutional Networks for Large-Scale Image Recognition. arXiv:1409.1556. 2014.
