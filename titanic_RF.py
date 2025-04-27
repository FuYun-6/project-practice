import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import ensemble, metrics

# 加载数据集
data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')

# 转为小写字母
data_train.columns = data_train.columns.str.lower()
data_test.columns = data_test.columns.str.lower()

pd.set_option('display.max_columns', None)
print(data_train.head())

# 查看数据集是否有空值
print(data_train.isnull().sum())
print(data_test.isnull().sum())

# 删除一些字段
drop_columns = ['cabin', 'ticket', 'passengerid']
data_train.drop(drop_columns, axis=1, inplace=True)
data_test.drop(drop_columns, axis=1, inplace=True)

# 数据清洗
data_all = pd.concat([data_train, data_test], ignore_index=True)

# 填充缺失值
data_all['age'] = data_all['age'].fillna(data_all['age'].median())
data_all['fare'] = data_all['fare'].fillna(data_all['fare'].median())
data_all['embarked'] = data_all['embarked'].fillna(data_all['embarked'].mode()[0])

# 检查是否还有空值
print(data_all.isnull().sum())

# 特征构建
# 1.家庭规模
data_all['family_size'] = data_all['sibsp'] + data_all['parch'] + 1
# 2.单身:1, 非单身：0
data_all['single'] = 1
data_all.loc[data_all['family_size'] > 1, 'single'] = 0
# 3.身份
data_all['title'] = data_all['name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]

# 合并少数类别为 'Rare'
title_counts = data_all['title'].value_counts()
rare_titles = title_counts[title_counts < 10].index  # 设定阈值
data_all['title'] = data_all['title'].replace(rare_titles, 'Rare')

# 将title转换为分类变量
title_mapping = {
    "mr": 0, "miss": 1, "mrs": 2, "master": 3, "dr": 4, "rev": 5,
    "sir": 6, "major": 7, "col": 8, "ms": 9, "lady": 10, "countess": 11,
    "don": 12, "jonkheer": 13, "mmme": 14, "mme": 15, "the": 16,
    "rare": 17
}
data_all['title'] = data_all['title'].map(title_mapping)

# 其他特征构建
data_all['fare_bin'] = pd.qcut(data_all['fare'], 4)
data_all['fare_bin'] = data_all['fare_bin'].apply(lambda x: (x.left + x.right) / 2)
data_all['age_bin'] = pd.qcut(data_all['age'].astype(int), 5, duplicates='drop')
data_all['age_bin'] = data_all['age_bin'].apply(lambda x: (x.left + x.right) / 2)

# 将embarked转换为分类变量
embarked_mapping = {'C': 0, 'Q': 1, 'S': 2}
data_all['embarked'] = data_all['embarked'].map(embarked_mapping)

# 将性别转换为数值：男性为0，女性为1
data_all['sex'] = data_all['sex'].map({'male': 0, 'female': 1})

# 分离训练集和测试集
data_train = data_all[:len(data_train)]
data_test = data_all[len(data_train):]

# 特征选择：选择对模型有用的特征
features = ['pclass', 'sex', 'age', 'fare', 'family_size', 'single', 'embarked', 'title', 'fare_bin', 'age_bin']
X_train = data_train[features]
y_train = data_train['survived']
X_test = data_test[features]

# 使用随机森林进行训练
rf_model = ensemble.RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 评估模型
# 预测及准确率
y_pred = rf_model.predict(X_test)
accuracy = metrics.accuracy_score(y_train, rf_model.predict(X_train))
print(f'准确率: {accuracy:.4f}')

# 分类报告
class_report = metrics.classification_report(y_train, rf_model.predict(X_train))
print('分类报告:')
print(class_report)

# 混淆矩阵
conf_matrix = metrics.confusion_matrix(y_train, rf_model.predict(X_train))
print('混淆矩阵:')
print(conf_matrix)

# 可视化混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


