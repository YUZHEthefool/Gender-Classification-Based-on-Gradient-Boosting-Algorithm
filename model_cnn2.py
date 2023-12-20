import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


class GenderDataset(Dataset):  # 定义一个继承自PyTorch Dataset的类，用于处理数据
    def __init__(self, X, y):  # 初始化函数，接收特征X和标签y
        self.X = X  # 特征
        self.y = y  # 标签

    def __len__(self):  # 定义数据集的长度
        return len(self.y)  # 返回标签的长度

    def __getitem__(self, idx): # 定义获取数据集中一个元素的方法
        return torch.tensor(self.X[idx], dtype=torch.float), torch.tensor(self.y[idx], dtype=torch.float)
        # 返回特征和标签的张量形式

data = pd.read_csv('Training_setdata.csv')  # 使用pandas读取CSV文件
X = data[['Weight', 'Height']].values    # 提取'Weight'和'Height'作为特征
y = data['Gender'].values.astype(float)  # 将标签转换为浮点数类型
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)   # 划分训练集和测试集

train_data = GenderDataset(X_train, y_train)    # 创建训练数据集
test_data = GenderDataset(X_test, y_test)   # 创建测试数据集


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 200)  # 增加神经元数量
        self.bn1 = nn.BatchNorm1d(200)
        self.dropout1 = nn.Dropout(0.5)  # 添加dropout层
        self.fc2 = nn.Linear(200, 100)
        self.bn2 = nn.BatchNorm1d(100)
        self.dropout2 = nn.Dropout(0.5)  # 添加dropout层
        self.fc3 = nn.Linear(100, 50)
        self.bn3 = nn.BatchNorm1d(50)
        self.dropout3 = nn.Dropout(0.5)  # 添加dropout层
        self.fc4 = nn.Linear(50, 25)
        self.bn4 = nn.BatchNorm1d(25)
        self.dropout4 = nn.Dropout(0.5)  # 添加dropout层
        self.fc5 = nn.Linear(25, 1)

    def forward(self, x):
        x = self.dropout1(F.relu(self.bn1(self.fc1(x))))  # 在激活函数后添加dropout
        x = self.dropout2(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout3(F.relu(self.bn3(self.fc3(x))))
        x = self.dropout4(F.relu(self.bn4(self.fc4(x))))
        x = torch.sigmoid(self.fc5(x))
        return x.squeeze()

model = Net().cuda()
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)  # 更改优化器为Adam
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')  # 更改学习率调度器为ReduceLROnPlateau
criterion = nn.BCELoss()    # 创建二元交叉熵损失函数
# best_val_loss = float('inf')  # 初始化最佳验证损失
# patience = 10  # 设置耐心参数，即在多少个epoch后验证损失还没有改善就停止训练
# patience_counter = 0  # 初始化耐心计数器

for epoch in range(100):
    model.train()  # 将模型设置为训练模式
    for inputs, labels in DataLoader(train_data, batch_size=32, shuffle=True):  # 使用数据加载器加载训练数据，批大小为32，每个epoch都打乱数据
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()   # 清零优化器的梯度
        outputs = model(inputs) # 通过模型计算输出
        loss = criterion(outputs, labels)  # 在这里移除了unsqueeze操作，计算输出
        loss.backward()     # 反向传播，计算梯度
        optimizer.step()    # 优化器更新模型参数
    model.eval()  # 将模型设置为评估模式
    with torch.no_grad():   # 关闭梯度计算
        val_loss = 0
        for inputs, labels in DataLoader(test_data, batch_size=32):  # 使用数据加载器加载测试数据，批大小为32
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)     # 通过模型计算输出
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    val_loss /= len(test_data)
    scheduler.step(val_loss)    # 学习率调度器更新学习率，传入验证损失
    # # 在每个epoch后检查验证损失
    # if val_loss < best_val_loss:
    #     best_val_loss = val_loss
    #     patience_counter = 0  # 如果验证损失改善，重置耐心计数器
    # else:
    #     patience_counter += 1  # 如果验证损失没有改善，增加耐心计数器
    #
    # # 如果耐心计数器达到耐心参数，停止训练
    # if patience_counter >= patience:
    #     print('Early stopping')
    #     break

#测试模型
model.eval()    # 将模型设置为评估模式
with torch.no_grad():   # 关闭梯度计算
    correct = 0     # 正确预测的数量
    total = 0   # 总的预测数量
    for inputs, labels in DataLoader(test_data, batch_size=32):# 使用数据加载器加载测试数据，批大小为32
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)     # 通过模型计算输出
        predicted = (outputs > 0.5).float()     # 将输出大于0.5的预测为1，小于等于0.5的预测为0
        total += labels.size(0)     # 更新总的预测数量
        correct += (predicted == labels).sum().item()   # 更新正确预测的数量
    print('Accuracy: {:.2f}%'.format(100 * correct / total))    # 打印准确率

torch.save(model.state_dict(), 'model_cnnnew.pth')    # 保存模型的参数