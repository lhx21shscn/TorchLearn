import torch
import torch.nn as nn
import torch.nn.functional as F
import d2l

def vgg_block(num_conv, in_channels, out_channels):
    layers = []
    for _ in range(num_conv):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

    return nn.Sequential(*layers)

def vgg(config):
    """
    网络的输入默认1 * 224 * 224,如果有变化，需要改写结构。
    :param config: vgg块的参数，形式为(num_conv, out_channels)
    :return: vgg网络
    """
    blocks = []
    in_channels = 1
    # 卷积部分
    for block in config:
        num_conv = block[0]
        out_channels = block[1]
        blocks.append(vgg_block(num_conv, in_channels, out_channels))
        in_channels = out_channels

    # 全连接部分
    return nn.Sequential(*blocks, nn.Flatten(),
                         nn.Linear(out_channels * 7 * 7, 4096), nn.Dropout(p=0.5), nn.ReLU(),
                         nn.Linear(4096, 4096), nn.Dropout(p=0.5), nn.ReLU(),
                         nn.Linear(4096, 10))
    # tips: 内存主要集中在第一个全连接层, 如果按照 main中的vgg参数结构,第一个全连接层拥有的参数个数是: (512 * 7 * 7 * 4096) = 102,760,448
    #       大小为 392M (float32)



if __name__ == '__main__':
    config = [(1, 64), (1, 128), (2, 256), (2, 512), (2, 512)]
    net = vgg(config)
    print(net)

