import torch
import torch.nn as nn
import torch.nn.functional as F
from DataLoader.Fashion_MNIST_Loader import Fashion_MNIST_Loader
from d2l import torch as d2l

# (1 * 28 * 28)

if __name__ == '__main__':

    mnist_train, mnist_test = Fashion_MNIST_Loader(256)
    print(mnist_train, mnist_test)

    LeNet = nn.Sequential(nn.Conv2d(1, 6, 5), nn.ReLU(), nn.MaxPool2d(2),
                          nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2),
                          nn.Flatten(),
                          nn.Linear(256, 120), nn.ReLU(),
                          nn.Linear(120, 64), nn.ReLU(),
                          nn.Linear(64, 10))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(LeNet.parameters(), 0.01)
    d2l.train_ch3(LeNet, mnist_train, mnist_test, loss_fn, 10, optimizer)