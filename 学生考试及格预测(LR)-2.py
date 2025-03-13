import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix


class LogisticRegression:
    def __init__(self, alpha, times):
        self.alpha = alpha
        self.times = times
        self.w_ = None
        self.loss_ = None

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)

        # 创建权重的向量，初始值为0长度比特征数多1.{多出来的是截距}
        self.w_ = np.zeros(1 + x.shape[1])

        # 创建损失列表用来保存每次迭代的损失值
        self.loss_ = []
        epsilon = 1e-6

        for i in range(self.times):
            z = np.dot(x, self.w_[1:]) + self.w_[0]
            # 计算概率值（结果判定为1的概率）
            p = self.sigmoid(z)
            p = np.clip(p, epsilon, 1 - epsilon)

            # 根据逻辑回归的代价函数，计算损失值
            # 逻辑回归的代价函数：
            # J(w) = -sum(yi * log(s(zi)) + (1 - yi) * log(1 - s(zi)) [i为从1到n，n为样本数量]
            cost = -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
            self.loss_.append(cost)

            # 调整权重值，根据公式调整为：权重(j列) = 权重(j列) + 学习率 * sum( (y - s(z)) * x(j))
            self.w_[0] += self.alpha * np.sum(y - p)
            self.w_[1:] += self.alpha * np.dot(x.T, y - p)

    def predict_proba(self, x):
        x = np.asarray(x)
        z = np.dot(x, self.w_[1:]) + self.w_[0]
        p = self.sigmoid(z)

        # 将预测结果转化为二维数组(结构)便于拼接
        p = p.reshape(-1, 1)

        # 将两个数组进行拼接，方向为横向拼接
        return np.concatenate([1 - p, p], axis=1)

    def predict(self, x):
        return np.argmax(self.predict_proba(x), axis=1)


# 数据读取
data = pd.read_csv('student_performance_dataset.csv')
# 分离特征和标签
x = data[['Study_Hours_per_Week', 'Attendance_Rate', 'Past_Exam_Scores' ]]
y = (data['Final_Exam_Score'] >= 60).astype(int)

# 数据预处理
x = x.fillna(x.mean())
scaler = StandardScaler()
x = scaler.fit_transform(x)

# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# 构建逻辑回归模型，可调整学习率和迭代次数
model = LogisticRegression(alpha=0.026, times=20000)

# 调用自定义类的 fit 方法训练模型
model.fit(x_train, y_train)

# 模型评估
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"准确率: {accuracy}")
print(f"精确率: {precision}")
print(f"F1值: {f1}")

# 绘制混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted results')
plt.ylabel('Actual results')
plt.title('Confusion matrix')
plt.show()