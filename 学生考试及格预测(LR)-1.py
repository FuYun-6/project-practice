import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix

#数据读取
data = pd.read_csv('Students_Grading_Dataset.csv')
#分离特征和标签
x = data[['Attendance (%)', 'Midterm_Score', 'Assignments_Avg', 'Quizzes_Avg', 'Participation_Score', 'Study_Hours_per_Week']]
y = (data['Final_Score'] >= 60).astype(int)



#数据预处理
x = x.fillna(x.mean())
scaler = StandardScaler()
x = scaler.fit_transform(x)

#划分数据集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
#构建逻辑回归模型
model = LogisticRegression(max_iter=1000)

#训练模型
model.fit(x_train, y_train)

#模型评估
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"准确率: {accuracy}")
print(f"精确率: {precision}")
print(f"F1值: {f1}")

#绘制混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted results')
plt.ylabel('Actual results')
plt.title('Confusion matrix')
plt.show()
