import time
import pandas as pd
import numpy as ny
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import ensemble, feature_selection, model_selection, metrics
from sklearn.preprocessing import LabelEncoder

#加载数据集
data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')

data_train.columns = data_train.columns.str.lower()
data_test.columns = data_test.columns.str.lower()

#查看数据集是否有空值
print(data_train.isnull().sum())
print(data_test.isnull().sum())

#删除一些字段
drop_columns = ['passengerid', 'cabin', 'ticket']
data_train.drop(drop_columns, axis = 1, inplace=True)
data_test.drop(drop_columns, axis = 1, inplace=True)

#合并两个数据集，进行数据清洗
data_all = pd.concat([data_train,data_test],sort=False)


# 数据清洗
data_all['age'] = data_all['age'].fillna(data_all['age'].median())
data_all['fare'] = data_all['fare'].fillna(data_all['fare'].median())
data_all['embarked'] = data_all['embarked'].fillna(data_all['embarked'].mode()[0])

print(data_all.isnull().sum())




