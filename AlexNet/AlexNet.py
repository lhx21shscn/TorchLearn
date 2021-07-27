import torch
import torch.nn as nn
from d2l import torch as d2l
from DataLoader.Fashion_MNIST_Loader import Fashion_MNIST_Loader

if __name__ == '__main__':
    # (224, 224, 1)
    AlexNet = nn.Sequential(nn.Conv2d(1, 96, kernel_size=11, stride=4), nn.ReLU(),
                            nn.MaxPool2d(kernel_size=3, stride=2),
                            nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
                            nn.MaxPool2d(kernel_size=3, stride=2),
                            nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
                            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
                            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
                            nn.MaxPool2d(kernel_size=3, stride=2), nn.Flatten(),
                            nn.Linear(6400, 4096), nn.ReLU(), nn.Dropout(p=0.5),
                            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=0.5),
                            nn.Linear(4096, 10))

    batch_size, lr, num_epoch = 128, 0.05, 10
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(AlexNet.parameters(), 0.05)
    # mnist_train, mnist_test = d2l.load_data_fashion_mnist(batch_size)
    mnist_train, mnist_test = Fashion_MNIST_Loader(batch_size, resize=224)
    print(mnist_train, mnist_test)
    d2l.train_ch3(AlexNet, mnist_train, mnist_test, loss_fn, 10, optimizer)