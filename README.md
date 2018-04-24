# pytorch-lr-explorer
A Jupyter notebook exploring various sophisticated techniques for working with learning rate in deep neural networks, including:
* A systematic method for estimating optimal learning rate settings by running a short training cycle throughout which the learning rate is continually increased, and then plotting the learning rate against loss or accuracy. The idea comes from [this paper](https://arxiv.org/pdf/1506.01186.pdf) by Leslie N. Smith.
* Time-based learning rate scheduling, and in particular cosine annealing with warm restarts. In this technique the learning rate is varied cyclically between certain boundaries in a cosine pattern. This technique is based on [this paper](https://arxiv.org/pdf/1608.03983.pdf) by Ilya Loshchilov & Frank Hutter.
* Snapshot ensembling, an efficient ensembling technique where before each warm restart a snapshot of the model is taken, and the last _M_ snapshots are combined as an ensemble. This technique comes from [this paper](https://arxiv.org/pdf/1704.00109.pdf) by Gao Huang, Yixuan Li, Geoff Pleiss, Zhuang Liu, John E. Hopcroft, Kilian Q. Weinberger.  

These strategies are demonstrated with respect to a relatively simple image classification problem (CIFAR10) using a basic resnet-style convolutional neural network created with [PyTorch](pytorch.org).
