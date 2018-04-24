import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import datasets, transforms

class CIFAR10Data:
    
    def __init__(self, batch_size):
        
        self.batch_size = batch_size

        self.training_set_transform = transforms.Compose(
                [transforms.RandomResizedCrop(32, scale=(0.9, 1.0)),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.training_set = datasets.CIFAR10(
                root='CIFAR10_data', train=True,
                transform=self.training_set_transform,
                download=True)
        self.training_set_loader = torch.utils.data.DataLoader(
                self.training_set, batch_size=self.batch_size,
                shuffle=True, num_workers=4)
    
        self.test_set_transform = transforms.Compose(
                [transforms.ToTensor(), 
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.test_set = datasets.CIFAR10(
                root='CIFAR10_data', train=False,
                transform=self.test_set_transform,
                download=True)
        self.test_set_loader = torch.utils.data.DataLoader(
                self.test_set, batch_size=self.batch_size,
                shuffle=False, num_workers=4)
        
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 
                        'dog', 'frog', 'horse', 'ship', 'truck')
        
    def display_batch(self, which_set='train'):
        if which_set == 'train':
            dataloader = iter(self.training_set_loader)
        elif which_set == 'test':
            dataloader = iter(self.test_set_loader)
        images, _ = next(dataloader)
        grid = torchvision.utils.make_grid(images, normalize=True)
        grid = np.transpose(grid.numpy(), (1, 2, 0))
        plt.imshow(grid)