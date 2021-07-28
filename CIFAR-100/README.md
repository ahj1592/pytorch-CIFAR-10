# pytorch-CIFAR-10
- pytorch CNN using CIFAR-100 dataset
- using data augmentation (described below)

# Summary
|model|CNN|
|:------:|:------:|
|Accuracy|**55%**|
|train size|250k|
|test size|10k|
|epoch|30|
|convergence|15|
|optimizer|Adam|
|scheduler|O|


# What I used
## Data Augmentation
Since each class has only 500 images in train dataset, I apply the data augmentation
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
