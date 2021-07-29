import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from torch.utils import data

def pre_process(data):
    # 处理缺失值
    data['Age'] = data['Age'].fillna(data['Age'].median())
    data['Embarked'] = data['Embarked'].fillna('S')

    # 处理str
    data.loc[data['Sex'] == 'male', 'Sex'] = 0
    data.loc[data['Sex'] == 'female', 'Sex'] = 1
    data.loc[data['Embarked'] == 'S', 'Embarked'] = 0
    data.loc[data['Embarked'] == 'C', 'Embarked'] = 1
    data.loc[data['Embarked'] == 'Q', 'Embarked'] = 2

    # 删除不需要的列
    predector = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    res = data.loc[:, predector].values.astype(np.float32)
    # 归一化
    res[:, 0] /= 3
    res[:, 2] /= 100
    res[:, 3] /= 8
    res[:, 4] /= 6
    res[:, 5] /= 220
    res[:, 6] /= 2

    if 'Survived' in data.columns:
        return res, data.loc[:, 'Survived'].values.astype(np.int64)
    return res

def train(net, loss_fn, optimizer, data_iter, num_epoch):
    net.train()
    for epoch in range(num_epoch):
        print('training epoch : ', epoch)
        net.train()
        for X, y in data_iter:
            # print(y[0].dtype)
            optimizer.zero_grad()
            output = net(X)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
        print('epoch ', epoch, ' acc: ', eval_acc(net, data_iter))

def eval_survived(net, test_iter):
    net.eval()
    res = np.zeros(418)

    for i in range(418):
        y = net(test_iter[i][None, :])
        if y[0][1] > y[0][0]:
            res[i] = 1

    return res.astype(np.uint8)




def eval_acc(net, data_iter):
    net.eval()
    cnt = 0
    for X, y in data_iter:
        output = net(X)
        # batch_size * 2
        cnt += torch.sum((output.argmax(dim=1) == y).float())
    return cnt / 891


# path
root = 'titanic'
train_path = os.path.join(root, 'train.csv')
test_path = os.path.join(root, 'test.csv')
res_path = os.path.join(root, 'res.csv')

# load data
titanic = pd.read_csv(train_path)
test_set = pd.read_csv(test_path)

# pre-processing
train_data, train_label = pre_process(titanic)
train_data = torch.tensor(train_data)
train_label = torch.tensor(train_label)
test = torch.tensor(pre_process(test_set))
data_set = data.TensorDataset(train_data, train_label)

# net-define
net = nn.Sequential(
    nn.Linear(7, 256), nn.ReLU(),
    nn.Linear(256, 128), nn.ReLU(),
    nn.Linear(128, 128), nn.ReLU(),
    nn.Linear(128, 128), nn.ReLU(),
    nn.Linear(128, 2)
)

# train
batch_size, lr, num_epoch = 3, 0.001, 10
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr, weight_decay=0.0001)
data_iter = data.DataLoader(data_set, batch_size=batch_size, shuffle=True)

train(net, loss_fn, optimizer, data_iter, num_epoch)

# predict and save
predict = eval_survived(net, test)

res = pd.DataFrame(predict, index=range(892, 1310), columns=['Survived'])
res.to_csv(res_path)