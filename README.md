# pytorch-lr-explorer
A Jupyter notebook exploring various sophisticated techniques for working with learning rate in deep neural networks, including:
* A systematic method for estimating optimal learning rate settings, in which the learning rate is constantly increased for a short duration of training, and then plotted against training loss or accuracy. This idea was introduced in [this paper](https://arxiv.org/abs/1506.01186) by Leslie N. Smith.
* Time-based learning rate scheduling, and in particular cosine annealing with warm restarts, where the learning rate is cyclically varied between high and low boundaries in a cosine pattern. This particular technique comes from [this paper](https://arxiv.org/abs/1608.03983.pdf) by Ilya Loshchilov and Frank Hutter.
* Snapshot ensembling, an extension to the above technique which involves taking a snapshot of the model after every cycle and using the last *M* snapshots as an ensemble. This technique is based on [this paper](https://arxiv.org/abs/1704.00109) by Gao Huang, Yixuan Li, Geoff Pleiss, Zhuang Liu, John E. Hopcroft and Kilian Q. Weinberger.

These strategies are demonstrated with respect to a relatively simple image classification problem (CIFAR10) using a simple resnet-style convolutional neural network created with [PyTorch](http://pytorch.org).
