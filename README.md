# pytorch-CIFAR-10
- pytorch CNN using CIFAR-10 dataset
- using data augmentation (described below)

# Summary
||CNN ver1|CNN ver2|CNN ver3|
|:-:|:------:|:------:|:------:|
|Accuracy|89% ~ 90%|89% ~ 90%|**91%**|
|train size|50k|250k|250k|
|test size|10k|10k|10k|
|epoch|50|50|30|
|convergence|30|15|17|
|optimizer|Adam|Adam|Adam|
|scheduler|-|-|ReduceLROnPlateau|

# Description of Model version
## CNN ver1
- data augmentation at download the dataset some probability
- data size: 50,000
- converge on epoch 30
- accuracy: 89% ~ 90%
- 
## CNN ver2 
- based on ver1
- remove 3 convolution layers
- data augmentation at dataloader
- data size: 250,000
- converge on epoch 15
- accuracy: 89% ~ 90%

## CNN ver3
- based on ver2
- remove 3 convolution layers
- remove 1 dropout
- reset dropout probability
- add `scheduler`
- converge on epoch 17
- accuracy: 91%


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
- dropout: 0.2(0.5 on ver3), only on Linear

## Train
- initialization: kaiming_uniform(a.k.a He initialization), normal, constant
- optimizer: Adam
- learning rate: 0.001
- scheduler: ReduceLROnPlateau (only ver3)
- loss: CrossEntropy
- epochs: 30 ~ 30
