# LMSER-pytorch

## Introduction

A pytorch implementation of LMSER, which is a bidirectional architecture with some built-in natures. In this project, we implement the Dual Connection Weight (DCW) property of LMSER. We implemented both LMSER with and without the pseudo inverse constrain, and compare their performance on reconstructing [MNIST](http://yann.lecun.com/exdb/mnist/) and [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) with AutoEncoder. We also add the supervised learning to the middle-layer embedding vector (16-d vector in our implementation), which aims to test the supervised learning effect on original reconstruction task.

## Prerequisites

* Linux, Ubuntu 16.04 or 18.04
* Python 3.6+
* Pytorch 1.4.0+
* CUDA 10.0+

## Installation

1. Create a conda virtual environment and activate it.

```
conda create -n lmser python=3.6
conda activate lmser
```

2. Install Pytorch and torchvision following the [official instruction](https://pytorch.org/).

3. Clone the LMSER-Pytorch repository.

```
git clone https://github.com/Simon-Fuhaoyuan/LMSER-pytorch
```

4. Install build requirements

```
pip install -r requirements.txt
```

## Usage

We offer both training and testing codes.

To train a model, please run

```
python train.py [-h] [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--lr LR] \
                [--mean MEAN] [--std STD] [--dataset DATASET] \
                [--weight_dir WEIGHT_DIR] [--image_dir IMAGE_DIR] [--cpu_only] \
                [--no_test] \
                model
```

For example, to train AutoEncoder, run

```
python train.py AutoEncoder --epochs 50 --batch_size 64 --dataset mnist
```

To test a model, run 
```
python test.py [-h] [--dataset DATASET] [--batch_size BATCH_SIZE] \
               [--mean MEAN] [--std STD] [--image_dir IMAGE_DIR] [--cpu_only] \
               model weight
```

For example, to test a trained model, whose weight model is saved at ```./weight_dir/AutoEncoder.pth```, run

```
python test.py AutoEncoder ./weight_dir/AutoEncoder.pth --dataset mnist
```

The usage of ```train_supervise.py``` and ```test_supervise.py``` is quite similar, but they will load the supervised version of model and train or test.

## Results

TODO
