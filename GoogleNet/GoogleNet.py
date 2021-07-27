import torch
import torch.nn as nn
from torch.nn import functional as F
from d2l import torch as d2l

class Inception(nn.Module):

    def __init__(self, in_channels, c1, c2, c3, c4):
        super(Inception, self).__init__()
        # 第一部分 1 * 1的卷积抽取特征
        self.conv1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 第二部分 1*1卷积降维 3*3抽取特征
        self.conv2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.conv2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 第三部分 1*1卷积降维 5*5抽取特征
        self.conv3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.conv3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 第四部分 3*3Maxpool 1*1卷积
        self.conv4_1 = nn.MaxPool2d(kernel_size=3, padding=1, stride=1)
        self.conv4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        res1 = F.relu(self.conv1_1(x))
        res2 = F.relu(self.conv2_2(F.relu(self.conv2_1(x))))
        res3 = F.relu(self.conv3_2(F.relu(self.conv3_1(x))))
        res4 = F.relu(self.conv4_2(self.conv4_1(x)))
        # 结果在通道维度上叠加
        return torch.cat((res1, res2, res3, res4), dim=1)


def GoogleNet():
    b1 = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    )
    b2 = nn.Sequential(
        nn.Conv2d(64, 64, kernel_size=1), nn.ReLU(),
        nn.Conv2d(64, 192, kernel_size=3, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    b3 = nn.Sequential(
        Inception(192, 64, (96, 128), (16, 32), 32),
        Inception(256, 128, (128, 192), (32, 96), 64),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    b4 = nn.Sequential(
        Inception(480, 192, (96, 208), (16, 48), 64),
        Inception(512, 160, (112, 224), (24, 64), 64),
        Inception(512, 128, (128, 256), (24, 64), 64),
        Inception(512, 112, (144, 288), (32, 64), 64),
        Inception(528, 256, (160, 320), (32, 128), 128),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    b5 = nn.Sequential(
        Inception(832, 256, (160, 320), (32, 128), 128),
        Inception(832, 384, (192, 384), (48, 128), 128),
        nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
    )
    net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))
    return net

if __name__ == '__main__':
    print(d2l.try_gpu())
    print(type(d2l.try_gpu()))
    batch_size, lr, num_epoch = 64, 0.05, 10
    net = GoogleNet()
    device = None
    mnist_train, mnist_test = d2l.load_data_fashion_mnist(batch_size, resize=224)
    d2l.train_ch6(net, mnist_train, mnist_test, num_epoch, lr, device=d2l.try_gpu())

    # pass
    # print(net)
    # data = torch.rand(2, 1, 224, 224)
    # print(data.shape)
    # for layer in net:
    #     data = layer(data)
    #     print(data.shape)

