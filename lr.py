import numpy as np


class CosineAnnealing:
    
    def __init__(self, optimizer, min_lr, max_lr, cycle_len, cycle_mult):
        self.i = 0
        self.optimizer = optimizer
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.cycle_len = cycle_len
        self.i_max = cycle_len - 1
        self.cycle_mult = cycle_mult
        self.lrs = []
        
    def get_lr(self):
        # linearly scale iteration to be between 0 and pi
        x = self.i / self.i_max * np.pi
        # take cosine of scaled iteration and linearly scale it 
        # to be between min_lr and max_lr
        lr = (self.max_lr-self.min_lr)/2 * (np.cos(x) + 1) + self.min_lr
        return lr
        
    def step(self):
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.lrs.append(lr)
        if self.i >= self.i_max:
            self.i = 0
            self.cycle_len *= self.cycle_mult
            self.i_max = self.cycle_len - 1
        else:
            self.i += 1


class Exponential:
    
    def __init__(self, optimizer, n, base_lr):
        self.i = 0
        self.optimizer = optimizer
        self.n = n
        self.base_lr = base_lr
        self.lrs = []
        
    def get_lr(self):
        lr = self.base_lr * self.n ** self.i
        return lr
    
    def step(self):
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.lrs.append(lr)
        self.i += 1