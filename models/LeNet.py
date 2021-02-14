from torch.nn import Module, Conv2d, AvgPool2d, Linear
import torch.nn.functional as F

class LeNet(Module):
  def __init__(self):
    super().__init__()
    self.conv1 = Conv2d(1, 6, (5,5), stride=1, padding=0)
    self.pool1 = AvgPool2d((2,2), stride=2, padding=0)
    self.conv2 = Conv2d(6, 16, (5,5), stride=1, padding=0)
    self.pool2 = AvgPool2d((2,2), stride=2, padding=0)
    self.linear1 = Linear(256, 120)
    self.linear2 = Linear(120, 84)
    self.linear3 = Linear(84, 10)

  def forward(self, x):
    x = F.tanh(self.conv1(x))
    x = self.pool1(x)
    x = F.tanh(self.conv2(x))
    x = self.pool2(x)
    x = x.view(x.shape[0], -1)
    x = F.tanh(self.linear1(x))
    x = F.tanh(self.linear2(x))
    x = self.linear3(x)
    return x