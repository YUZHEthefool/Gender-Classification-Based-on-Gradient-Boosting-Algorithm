import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# 加载数据
df = pd.read_csv('Training_setdata.csv')

# 数据预处理
df['Height'] = df['Height'] / df['Height'].max()
df['Weight'] = df['Weight'] / df['Weight'].max()

# 划分训练集和测试集
train_df, test_df = train_test_split(df, test_size=0.3)

class GenderDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data.iloc[idx, 1:].values, dtype=torch.float), torch.tensor(self.data.iloc[idx, 0], dtype=torch.long)

train_dataset = GenderDataset(train_df)
test_dataset = GenderDataset(test_df)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

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

model = GenderClassifier()
model = model.to('cuda')

optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 定义优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# 训练模型
for epoch in range(100):
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to('cuda')
        labels = labels.to('cuda')

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

# 测试模型
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        inputs = inputs.to('cuda')
        labels = labels.to('cuda')

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy: %d %%' % (100 * correct / total))

torch.save(model.state_dict(), 'model_cnn.pth')