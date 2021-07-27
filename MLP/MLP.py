import torch
from torch import nn
from d2l import torch as d2l
from torch.nn import functional as F
class MLPnet(nn.Module):

    def __init__(self):
        super(MLPnet, self).__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(784, 256)
        self.linear2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.linear1(self.flatten(x)))
        x = self.linear2(x)
        return x

net = MLPnet()
batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=lr)

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)