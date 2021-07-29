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


root = ""
train_file = os.path.join(root, 'train.csv')
test_file = os.path.join(root, 'test.csv')
save_path = 'models'


def check_image():
    """
    检查图片是否全部是(224, 224, 3)
    """
    for i in range(0, 27153):
        img_path = os.path.join(file_path, str(i) + '.jpg')
        img = Image.open(img_path)
        if img.size != (224, 224) or img.mode != 'RGB':
            return False
    return True

def pre_process(df, dictionary=None):
    """
    如果没有传入字典，将自动生成：
        对传入的df的label列用整数替换种类字符
    如果传入字典，将使用已有的字典：
        对传入的df的label列用种类字符代替整数
    """
    if dictionary is not None:
        # 用在已经生成了测试结果，把预测结果转换为种类名称时。
        for id, kinds in dictionary.items():
            if isinstance(kinds, int):
                continue
            df.loc[df['image'] == id, 'label'] = kinds
        return df
    else:
        # 用在将训练集dataframe的label列转换成整数，然后生成字典。
        cnt = 0
        dictionary = {}
        for i in df['label'].unique():
            dictionary[i] = cnt
            dictionary[cnt] = i
            cnt += 1
        for i in df['label'].unique():
            df.loc[df['label'] == i, 'label'] = dictionary[i]
        return df, dictionary

class LeavesDataset(Dataset):

    def __init__(self, file_path, df=None, mode='train'):
        self.file_path = file_path
        self.mode = mode
        self.transform = [transforms.ToTensor()]

        if mode == 'train':
            self.img_label = np.zeros(18353, dtype=np.int64)
            self.img_label[:] = df.iloc[:, 1].values
            self.transform.insert(0, transforms.RandomHorizontalFlip(p=0.5))

        self.transform = transforms.Compose(self.transform)

    def __len__(self):
        if self.mode == 'train':
            return 18353
        else:
            return 8800

    def __getitem__(self, index):

        # 读取图像
        if self.mode == 'test':
            index += 18353
        img_name = str(index) + '.jpg'
        img_path = os.path.join(self.file_path, img_name)
        img = Image.open(img_path)

        # 图像转换(增广)
        img = self.transform(img)

        if self.mode == 'test':
            return img
        else:
            return img, self.img_label[index]

# 是否要冻住模型的前面一些层
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        model = model
        for param in model.parameters():
            param.requires_grad = False

# resnet50模型
def res_model(num_classes, feature_extract = False, pretrained=True):

    net_ft = resnet50(pretrained=pretrained)
    set_parameter_requires_grad(net_ft, feature_extract)
    num_ftrs = net_ft.fc.in_features
    net_ft.fc = nn.Linear(num_ftrs, num_classes)

    return net_ft

def train(net, loss_fn, optimizer, train_iter, num_epoch, device):

    net.to(device)
    len_iter = len(train_iter)

    for epoch in range(num_epoch):

        print('training epoch: ', epoch)
        net.train()
        acc_sum = 0.0
        loss_sum = 0.0
        for X, y in tqdm(train_iter):

            optimizer.zero_grad()
            X = X.to(device)
            y = y.to(device)
            output = net(X)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
            acc = (output.argmax(dim=1) == y).float().mean()

            acc_sum += acc
            loss_sum += loss
        train_loss = loss_sum / len_iter
        train_acc = acc_sum / len_iter

        print(f"[ Train | {epoch + 1:03d}/{num_epoch:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        model_path = os.path.join(save_path, 'model_epoch_{}.pth'.format(epoch))
        torch.save(net.state_dict(), model_path)

if __name__ == "__main__":

    # 超参数
    batch_size, lr, weight_decay, num_epoch = 8, 2e-4, 0.001, 60

    # device
    device = 'cuda:0'

    # 网络 resnet50
    net = res_model(176, pretrained=True, feature_extract=False)

    # 数据预处理
    train_label = pd.read_csv(train_file)

    train_label, dictionary = pre_process(train_label)
    print(dictionary)
    # 数据集
    file_path = 'images'
    train_set = LeavesDataset(file_path, train_label, mode='train')
    train_iter = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=5)

    # train
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    train(net, loss_fn, optimizer, train_iter, num_epoch, device)









