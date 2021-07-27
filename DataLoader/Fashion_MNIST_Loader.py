import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
d2l.use_svg_display()

def get_dataloader_workers():
    # 设定读取的线程数
    """
    可以利用下面程序测试时间，以选择最佳的线程数去读取。
    timer = d2l.Timer()
    for X, y in train_iter:
        pass
    f'{timer.stop():.2f} sec'
    """
    return 4


def Fashion_MNIST_Loader(batch_size, resize=None, download=False):
    trans = [transforms.ToTensor()]
    if resize is not None:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root="../data",
                                                    train=True,
                                                    transform=trans,
                                                    download=download)
    mnist_test = torchvision.datasets.FashionMNIST(root="../data",
                                                   train=False,
                                                   transform=trans,
                                                   download=download)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))

if __name__ == '__main__':

    mnist_train, mnist_test = Fashion_MNIST_Loader(4, (224, 224))
    print(mnist_test, mnist_train)
    print(len(mnist_train), len(mnist_test))
    for X, y in mnist_test:
        print(X.shape)
        break
    # error
    # X, y = mnist_train[0]
    # print(X, y)