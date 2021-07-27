# 思想:完全不要全连接层
import torch
import torch.nn as nn
from d2l import torch as d2l

def nin_block(in_channels, out_channels, kernel_size, padding, stride):

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU()
    )

def nin():

    return nn.Sequential(
        nin_block(1, 96, kernel_size=11, stride=4, padding=0),
        nn.MaxPool2d(3, stride=2),
        nin_block(96, 256, kernel_size=5, stride=1, padding=2),
        nn.MaxPool2d(3, stride=2),
        nin_block(256, 384, kernel_size=3, stride=1, padding=1),
        nn.MaxPool2d(3, stride=2), nn.Dropout(0.5),
        nin_block(384, 10, kernel_size=3, stride=1, padding=1),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten()
    )

if __name__ == '__main__':
    net = nin()
    print(net)

    # 展示各层网络的输出形状
    data = torch.rand((1, 1, 224, 224))
    print(data.shape)
    for layer in net:
        data = layer(data)
        print(data.shape)