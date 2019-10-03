# RED-NN
Source code for Rotation Equivariant Neural Network

## Dataset
This repository contains the dataset used for training and validation. [Up-right oriented training set](../blob/master/MNIST_UR_train.npz) contains 60,000 MNIST images in up-right orientation. [Randomly rotated validation set](../blob/master/MNIST_RR_test.npz) contains 10,000 randomly oriented images. 


## Layer definition
The [rotational invariant layer](../blob/master/layer_definition.py) definition based on steerable filters is declared in this file. It contains the possibility of adding more filter ensembles but we only used one.

## Network architecture
The [RED-NN architecture](../blob/master/layer_definition.py) is contained in this file. This file is ready for training with other datasets.

For more information contact: **r.rodriguez@esiee.fr**

### Authors:
Rosemberg Rodriguez Salas, Petr Dokladal, Eva Dokladalova
