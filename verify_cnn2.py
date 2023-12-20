import torch
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from homework1.model_cnn2 import model
from homework1.text_cnn2 import test_data_new

# 加载模型参数
model.load_state_dict(torch.load('model_cnnnew.pth'))

# 将模型移动到 GPU 上
model.cuda()

# 测试模型并收集预测结果和真实标签
model.eval()
predictions, true_labels = [], []
with torch.no_grad():
    for inputs, labels in DataLoader(test_data_new, batch_size=32):
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)
        predicted = (outputs > 0.5).float()
        predictions.extend(predicted.tolist())
        true_labels.extend(labels.tolist())

# 计算精确度、召回率和 F1 分数
precision = precision_score(true_labels, predictions)
recall = recall_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions)
print('Precision: {:.2f}, Recall: {:.2f}, F1 Score: {:.2f}'.format(precision, recall, f1))

# 绘制 ROC 曲线和计算 AUC 值
fpr, tpr, _ = roc_curve(true_labels, predictions)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('DNN Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.rcParams['font.sans-serif']=['SimHei']
plt.figtext(0.5, 0.01, '图9', ha='center')
plt.show()

# 生成混淆矩阵
cm = confusion_matrix(true_labels, predictions)
print('Confusion Matrix:\n', cm)


# 检查过拟合和欠拟合（这需要训练过程中的损失和准确率数据，如果没有，这部分代码需要跳过）

# 交叉验证（由于PyTorch模型和Sklearn的交叉验证不兼容，这部分代码需要跳过）