# pytorch-lr-explorer
A Jupyter notebook exploring various sophisticated techniques for working with learning rate in deep neural networks, including:
* A systematic method for estimating optimal learning rate settings, where the learning rate is constantly increased for a short duration of training, and then plotted against training loss or accuracy. This idea was introduced in [this paper](https://arxiv.org/pdf/1506.01186.pdf) by Leslie N. Smith.
* Time-based learning rate scheduling, and in particular cosine annealing with warm restarts, where the learning rate is cyclically varied between high and low boundaries in a cosine pattern. This particular technique comes from [this paper](https://arxiv.org/pdf/1608.03983.pdf) by Ilya Loshchilov & Frank Hutter.
* Snapshot ensembling, an extension to the above technique where a snapshot is taken before every restart and the last _M_ snapshots are used as an ensemble. This technique is based on [this paper](https://arxiv.org/pdf/1704.00109.pdf) by Gao Huang, Yixuan Li, Geoff Pleiss, Zhuang Liu, John E. Hopcroft, Kilian Q. Weinberger.

These strategies are demonstrated with respect to a relatively simple image classification problem (CIFAR10) using a simple resnet-style convolutional neural network created with [PyTorch](http://pytorch.org).
