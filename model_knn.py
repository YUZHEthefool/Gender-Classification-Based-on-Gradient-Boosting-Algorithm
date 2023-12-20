import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

# 读取数据
data = pd.read_csv('Training_setdata.csv')

# 分割特征和标签
X = data[['Height', 'Weight']]
y = data['Gender']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建模型
knn = KNeighborsClassifier(n_neighbors=3)

# 训练模型
knn.fit(X_train, y_train)

# 预测
y_pred = knn.predict(X_test)

# 计算精确度、召回率和F1分数
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print('Accuracy: ', accuracy)
print('Recall: ', recall)
print('F1 score: ', f1)

# 生成混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix: \n', cm)

# 计算ROC曲线和AUC值
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('KNN Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.rcParams['font.sans-serif']=['SimHei']
plt.figtext(0.5, 0.01, '图3', ha='center')
plt.show()

# 进行交叉验证
scores = cross_val_score(knn, X, y, cv=5)
print('Cross-validation scores: ', scores)