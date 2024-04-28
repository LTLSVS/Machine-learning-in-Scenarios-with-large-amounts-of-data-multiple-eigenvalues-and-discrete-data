import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import datasets
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve
from imblearn.over_sampling import SMOTE
from collections import Counter

def A_N1(x):
    if x["COL3"] == 'A':
        return 1
    else:
        return 0
def A_N2(x):
    if x["COL4"] == 'A':
        return 1
    else:
        return 0
def A_N3(x):
    if x["COL5"] == 'A':
        return 1
    else:
        return 0

df = pd.DataFrame()
df = pd.read_csv('/work/data/train_base_data.csv',usecols=["CHANNEL_A","COL1","COL2","COL3","COL4","COL5","COL6","COL7","COL8","COL9","COL10","COL11","COL15","COL16","COL17","COL18"])
df_test= pd.read_csv('/work/data/test_data.csv',usecols=["COL1","COL2","COL3","COL4","COL5","COL6","COL7","COL8","COL9","COL10","COL11","COL15","COL16","COL17","COL18"])

df=df.dropna()
df["COL3"] = df.apply(A_N1,axis=1)
df["COL4"] = df.apply(A_N2,axis=1)
df["COL5"] = df.apply(A_N3,axis=1)
df_test["COL3"] = df_test.apply(A_N1,axis=1)
df_test["COL4"] = df_test.apply(A_N2,axis=1)
df_test["COL5"] = df_test.apply(A_N3,axis=1)
df["COL15"] = df.apply(lambda x: 3 * x["COL18"] - 3 * x["COL15"], axis=1)
df_test["COL15"] = df_test.apply(lambda x: 3 * x["COL18"] - 3 * x["COL15"], axis=1)

df = df.drop("COL18",axis=1)
df_test = df_test.drop("COL18",axis=1)

df_test = df_test.fillna(0)

X = df.loc[:, ["COL1","COL2","COL3","COL4","COL5","COL6","COL7","COL8","COL9","COL10","COL11","COL15","COL16","COL17"]]# 特征
y = df.loc[:, "CHANNEL_A"] # 目标变量

X_train = X
y_train = y
X_test = df_test.loc[:, ["COL1","COL2","COL3","COL4","COL5","COL6","COL7","COL8","COL9","COL10","COL11","COL15","COL16","COL17"]]


svm = SVC (kernel='linear', max_iter=10000)
# 训练模型
svm.fit(X_train, y_train)
# 在测试集上进行预测
y_pred = svm.predict(X_test)

# 输出结果
print("Predicted values:", y_pred)
print(Counter(y_pred))
print(Counter(y))
# 将预测结果保存为 DataFrame
predictions_a = pd.DataFrame({'CHANNEL_A': y_pred})
# 保存为 CSV 文件
predictions_a.to_csv('predicted_A.csv', index=False)