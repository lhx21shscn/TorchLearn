import torch
import torch.nn as nn
from torch.utils import data
import time
from d2l import torch as d2l

"""
一种低效的实现：
index = (torch.randn(X.shape) > p)
X[index] = 0
X *= 1 - p
return X
这样随机抽样非常蠢，torch有专门的函数进行抽样选择，比如numpy中的choice函数。
这只是一个例子去说明：抽样对cpu gpu是不友好的，很慢。应该尽量选取乘法去实现。
"""
def dropout_layer(X, p):
    if p == 1.0:
        return torch.zeros_like(X)
    if p == 0.0:
        return X
    mat = (torch.randn(X.shape) > p).float()
    return mat * X / (1 - p)

if __name__ == '__main__':
    pass


