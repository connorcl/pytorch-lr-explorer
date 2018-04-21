import torch
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from hw import use_gpu, FloatTensor, LongTensor
from lr import Exponential


class ConvBnLayer(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

    
class ResLayer(ConvBnLayer):
    
    def forward(self, x):
        return x + super().forward(x)

    
class CNN(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBnLayer(3, 32, 5, 1, 2)
        self.layer1 = nn.Sequential(
                ConvBnLayer(32, 64, 3, 2 ,1),
                ResLayer(64, 64, 3, 1, 1),
                ResLayer(64, 64, 3, 1, 1))
        self.layer2 = nn.Sequential(
                ConvBnLayer(64, 128, 3, 2, 1),
                ResLayer(128, 128, 3, 1, 1),
                ResLayer(128, 128, 3, 1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d(4)
        self.fc = nn.Linear(2048, 10)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = self.dropout(x.view(x.size(0), -1))
        x = self.fc(x)
        return x


class Model:
    
    def __init__(self, dataset, scheduler, scheduler_params):
        self.dataset = dataset
        self.model = CNN()
        self.optimizer = optim.Adam(self.model.parameters(), 0.0002)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = scheduler(self.optimizer, *scheduler_params)
        if use_gpu:
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()
    
    def _forward(self, data):
        inputs, targets = data
        inputs = Variable(inputs.type(FloatTensor))
        targets = Variable(targets.type(LongTensor))
        outputs = self.model(inputs)
        _, predictions = torch.max(outputs, 1)
        loss = self.criterion(outputs, targets)
        acc = (predictions == targets).sum().data[0] / self.dataset.batch_size
        return outputs, predictions, loss, acc
        
    def train(self, epochs=10):
        for epoch in range(1, epochs+1):
            running_loss = 0.0
            running_acc = 0.0
            i = 1
            with tqdm(self.dataset.training_set_loader, 
                      desc="Epoch %d/%d" % (epoch, epochs)) as t:
                for data in t:
                    self.optimizer.zero_grad()
                    outputs, predictions, loss, acc = self._forward(data)
                    loss.backward()
                    self.scheduler.step()
                    self.optimizer.step()
                    running_loss += loss.data[0]
                    running_acc += acc
                    t.set_postfix(loss=running_loss/i, acc=running_acc/i)
                    i += 1
            running_test_loss = 0.0
            running_test_acc = 0.0
            i = 1
            with tqdm(self.dataset.test_set_loader,
                      desc="Test set") as t:
                for data in t:
                    outputs, predictions, loss, acc = self._forward(data)
                    running_test_loss += loss.data[0]
                    running_test_acc += acc
                    t.set_postfix(loss=running_test_loss/i,
                                  acc=running_test_acc/i)
                    i += 1
    
    def save(self, savefile):
        torch.save({'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()},
                        savefile)
        print("Progress saved!")
        
    def load(self, savefile):
        state = torch.load(savefile)
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        print("Progress loaded!")


class LRFinder(Model):
    
    def __init__(self, dataset, n=1.01, base_lr=5e-6):
        super().__init__(dataset, Exponential, (n, base_lr))
        self.lrs = self.scheduler.lrs
        self.mv_avg_losses = []
        
    def train(self):
        self.lr_find()
                    
    def lr_find(self, epochs=1):
        
        losses = []
        
        for epoch in range(1, epochs+1):
            running_loss = 0.0
            running_acc = 0.0
        
            i = 1
            with tqdm(self.dataset.training_set_loader, 
                      desc="Epoch %d/%d" % (epoch, epochs)) as t:
                for data in t:
                    self.optimizer.zero_grad()
                    _, _, loss, acc = self._forward(data)
                    loss.backward()
                    self.scheduler.step()
                    self.optimizer.step()
                    running_loss += loss.data[0]
                    running_acc += acc
                    losses.append(loss.data[0])
                    n = min(32, len(losses))
                    s = sum(losses[-n:])
                    mv_avg_loss = s / n
                    self.mv_avg_losses.append(mv_avg_loss)
                    t.set_postfix(loss=running_loss/i,
                                  acc=running_acc/i, lr=self.scheduler.lrs[-1])
                    if mv_avg_loss > self.mv_avg_losses[min(31, len(self.mv_avg_losses)-1)] * 1.35:
                        break
                    i += 1
    
    def plot(self):
        plt.semilogx(self.lrs, self.mv_avg_losses)
        plt.show()

