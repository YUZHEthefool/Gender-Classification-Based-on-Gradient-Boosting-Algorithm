import pandas as pd
import torch

class GenderClassifier(torch.nn.Module):
    def __init__(self):
        super(GenderClassifier, self).__init__()
        self.linear1 = torch.nn.Linear(2, 100)
        self.bn1 = torch.nn.BatchNorm1d(100)
        self.linear2 = torch.nn.Linear(100, 50)
        self.bn2 = torch.nn.BatchNorm1d(50)
        self.linear3 = torch.nn.Linear(50, 2)

    def forward(self, inputs):
        outputs = torch.nn.functional.leaky_relu(self.bn1(self.linear1(inputs)))
        outputs = torch.nn.functional.leaky_relu(self.bn2(self.linear2(outputs)))
        return self.linear3(outputs)

def predict_gender(height, weight, df):
    # 加载模型
    model = GenderClassifier()
    model.load_state_dict(torch.load('model_cnn2.pth'))
    model = model.to('cuda')
    model.eval()

    # 将输入数据转换为张量，并进行归一化
    inputs = torch.tensor([height, weight]).float().cuda()
    inputs[0] /= df['Height'].max()  # 使用df来获取身高的最大值
    inputs[1] /= df['Weigh'].max()  # 使用df来获取体重的最大值

    # 将输入数据增加一个维度，以匹配模型的输入形状
    inputs = inputs.unsqueeze(0)

    # 使用模型进行预测
    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)

    # 返回预测结果
    return '女' if predicted.item() == 0 else '男'

# 加载数据
df = pd.read_csv('data.csv')

# 数据预处理
df['Height'] = df['Height'] / df['Height'].max()
df['Weigh'] = df['Weigh'] / df['Weigh'].max()

# 获取用户的输入
height = float(input('请输入你的身高（厘米）：')) / 100  # 假设身高单位是米
weight = float(input('请输入你的体重（千克）：'))

# 预测性别并打印结果
print('Predicted gender:', predict_gender(height, weight, df))