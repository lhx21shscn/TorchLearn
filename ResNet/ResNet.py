import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import torchvision.models as models
# 类名均是参考 PyTorch源码实现，PyTorch源码需要实现所有类型的ResNet，这里我仅仅实现resnet50所以简化了部分内容。

"""
实现参考：
https://arxiv.org/pdf/1512.03385.pdf
还有torchvision.models.resnet
"""

"""
残差块Residual在：
pytorch的ResNet实现中,分为BasicBlock,Bottleneck两种。
BasicBlock: 3*3->3*3
Bottleneck: 1*1->3*3->1*1
"""

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, downsample=None, strides=1):

        super(Bottleneck, self).__init__()
        self.downsample = downsample

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=strides, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        identity = x

        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu(y)
        y = self.conv3(y)
        y = self.bn3(y)

        if self.downsample is not None:
            identity = self.downsample(identity)
        y += identity
        y = self.relu(y)
        return y

class ResNet(nn.Module):

    def __init__(self, block, layers):
        super(ResNet, self).__init__()
        self.block = block

        # 所有 resnet的前端是一模一样的,输出的通道数都是 64
        # self.in_channels 会随着_make_layer的进行来变换。
        # 以resnet50为例： 64->4*64->4*128->4*256->4*521
        # 变换的规律是根据_make_layers_的channels参数变换随之改变
        self.in_channels = 64
        # 如果是灰度图像这里把 3->1
        self.head = nn.Sequential(
            nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # 如果图像是3 * 224 * 224的, 经过head后大小为 64 * 56 * 56

        self.layer1 = self._make_layers_(Bottleneck, 64, layers[0], 1)
        self.layer2 = self._make_layers_(Bottleneck, 128, layers[1], 2)
        self.layer3 = self._make_layers_(Bottleneck, 256, layers[2], 2)
        self.layer4 = self._make_layers_(Bottleneck, 521, layers[3], 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(521*block.expansion, 10)

    def _make_layers_(self, block, channels, num_block, stride):
        """
        :param block:        Union[BasicBlock,Bottleneck]
        :param channels:     通道数的基准，如果并不是真正的输出通道数，或者输入通道数。
                             block的输入通道数为 self.in_channels, 输出通道数为 block.expansion * channels
                             ps：Bottleneck.expansion = 4, BasicBlock.expansion = 1
        :param num_block:    block的数量
        :param stride:       步长, 如果stride==2那么 (H, W) -> (H/2, W/2)
        stride只会应用在第一个block里，在一开始达到降维的效果。

        """
        downsample = None

        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, channels, downsample=downsample, strides=stride))
        # 根据两种block的性质，在第一块降采样后，其余块不需要
        self.in_channels = channels * block.expansion
        for _ in range(1, num_block):
            layers.append(block(self.in_channels, channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        y = self.head(x)
        # print(y.shape)
        y = self.layer1(y)
        # print(y.shape)
        y = self.layer2(y)
        # print(y.shape)
        y = self.layer3(y)
        # print(y.shape)
        y = self.layer4(y)
        # print(y.shape)
        y = self.avgpool(y)
        # print(y.shape)
        y = y.reshape(-1, 521*self.block.expansion)
        y = self.fc(y)
        # print(y.shape)
        return y

def resnet_50():
    return ResNet(Bottleneck, [3, 4, 6, 3])

def resnet_101():
    return ResNet(Bottleneck, [3, 4, 23, 3])

def resnet_152():
    return ResNet(Bottleneck, [3, 8, 36, 3])

if __name__ == "__main__":
    net = resnet_50()
    net1 = models.resnet50()

    print(len(net.state_dict()), len(net1.state_dict()))
    # 从参数数量的完全相同可以看出其实和PyTorch实现的是应该一样的。
    # 如果某一天心血来潮要改成和PyTorch一模一样可以把state_dict输出然后对着一个一个佐证。
    for layer in net.state_dict():
        print(layer)
    for layer in net1.state_dict():
        print(layer)