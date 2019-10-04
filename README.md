# RED-NN
Source code for Rotation Equivariant Neural Network

## Dataset
This repository contains the dataset used for training and validation. [Up-right oriented training set](../blob/master/MNIST_UR_train.npz) contains 60,000 MNIST images in up-right orientation. [Randomly rotated validation set](../blob/master/MNIST_RR_test.npz) contains 10,000 randomly oriented images. 


## Layer definition
The [rotational invariant layer](../blob/master/layer_definition.py) definition based on steerable filters is declared in this file. It contains the possibility of adding more filter ensembles but we only used one.

## Network architecture
The [RED-NN architecture](../blob/master/layer_definition.py) is contained in this file. This file is ready for training with other datasets.

## Pre-trained models
Two pre-trained models are included in this repository. [Up-right trained model](../blob/master/URT_REDNN_16.h5) and [Randomly rotated model](../blob/master/RRT_REDNN_16.h5). This models can be loaded and tested with [this file](../blob/master/load_model.py).


For more information contact: **r.rodriguez@esiee.fr**

## Requirements
```
python = 3.6.9
tensorflow = 1.13.1 (Tested on tensorflow-gpu version 1.13.1)
matplotlib = 3.0.3
sklearn = 0.21.3
scipy = 1.2.1 (Recent version does not include rotate function)
```
An Anaconda environment can be created with the provided [yml file](../blob/master/environment.yml) and the command:
```
conda env create -f environment.yml
```

### Authors:
Rosemberg Rodriguez Salas, Petr Dokladal, Eva Dokladalova
