# pytorch-CIFAR-10
- pytorch CNN using CIFAR-10 dataset
- using data augmentation (described below)

# Description of Model version
## CNN ver1
- data augmentation at download the dataset some probability
- data size: 50,000
- converge on epoch 30
- accuracy: 89% ~ 90%
## CNN ver2 
- data augmentation at dataloader
- data size: 250,000
- converge on epoch 15
- accuracy: 89% ~ 90%

|---|CNN ver1|CNN ver2|CNN ver3|
|-|:------:|:------:|:------:|
|epoch|50|50|30|
|optimizer|Adam|
|scheduler|-|-|ReduceLROnPlateau|


# What I used
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
