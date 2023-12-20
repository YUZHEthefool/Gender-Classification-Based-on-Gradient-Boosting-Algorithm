import torch
import pandas as pd
from torch.utils.data import DataLoader

from homework1.model_cnn2 import Net, GenderDataset

# 加载模型
model = Net().cuda()
model.load_state_dict(torch.load('model_cnnnew.pth'))

# 加载新的测试集
test_data_new = pd.read_csv('Training_setdata.csv')
X_new = test_data_new[['Weight', 'Height']].values
y_new = test_data_new['Gender'].values.astype(float)

# 创建新的测试数据集对象
test_data_new = GenderDataset(X_new, y_new)

# 测试模型
model.eval()  # 将模型设置为评估模式
with torch.no_grad():  # 关闭梯度计算
    correct = 0  # 正确预测的数量
    total = 0  # 总的预测数量
    for inputs, labels in DataLoader(test_data_new, batch_size=32):  # 使用数据加载器加载测试数据，批大小为32
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)  # 通过模型计算输出
        predicted = (outputs > 0.5).float()  # 将输出大于0.5的预测为1，小于等于0.5的预测为0
        total += labels.size(0)  # 更新总的预测数量
        correct += (predicted == labels).sum().item()  # 更新正确预测的数量
    print('Accuracy: {:.2f}%'.format(100 * correct / total))  # 打印准确率
