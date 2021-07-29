import numpy
import torch
import torch.nn as nn
from torchvision.models.resnet import *
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from train import res_model, LeavesDataset, pre_process

# 可以修改的参数是epoch, epoch=x代表选取第x次迭代产生的模型
root = ''
train_file = os.path.join(root, 'train.csv')
test_file = os.path.join(root, 'test.csv')
net_path, epoch = 'models', 0
net_path = os.path.join(net_path, 'model_epoch_' + str(epoch) + '.pth')
save_file = './submission.csv'

def test(net, test_iter, device, dictionary):
    predict = []

    net.eval()
    net.to(device)

    for X in tqdm(test_iter):
        with torch.no_grad():
            output = net(X.to(device))
        predict.append(torch.argmax(output))

    pred = []
    for id in predict:
        pred.append(dictionary[id])

    return pred



if __name__ == '__main__':

    # 网络
    net = res_model(176)
    net.load_state_dict(torch.load(net_path))
    device = 'cuda:0'

    # test
    train_label = pd.read_csv(train_file)
    train_label, dictionary = pre_process(train_label)
    test_df = pd.read_csv(test_file)

    test_set = LeavesDataset('images', mode='test')
    test_iter = DataLoader(test_set, shuffle=False, batch_size=8, num_workers=5)
    predict = test(net, test_iter, device, dictionary)

    test_df['label'] = pd.Series(predict)
    submission = pd.concat([test_df['image'], test_df['label']], axis=1)
    submission.to_csv(save_file, index=False)
    print("Done!!!!!!!!!!!!!!!!!!!!!!!!!!!")

