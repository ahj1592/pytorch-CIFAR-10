# pytorch-CIFAR-10
practice for pytorch CNN

# Accuracy: about 90%

# What I used in
## Data Augmentation
- random horizontal flip
- random rotation
- random affine
- color jitter
- normalization

## Model
- based on CNN, VGG16
- kernel size: 3
- use padding to keep feature size
- use batch normalization
- max pooling: (2, 2)
- activation function: ReLU, LogSoftmax
- dropout: 0.2, only on Linear

## Train
- initialization: kaiming_uniform(a.k.a He initialization), normal, constant
- optimizer: Adam
- loss: CrossEntropy
- epochs: 50
