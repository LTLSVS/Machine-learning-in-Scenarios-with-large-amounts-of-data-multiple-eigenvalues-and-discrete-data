import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve
from imblearn.over_sampling import SMOTE


def f(x):
    if x["MSSubClass"]>=50:
        return 1
    else:
        return 0

# 加载数据集
train = pd.read_csv('train.csv',usecols=[0,1,2,3,17,18,19])
train.dropna(inplace=True)
train["MSSubClass"] = train.apply(f,axis=1)
print(train.head(3))
X = train.loc[:1000, ["OverallCond","OverallQual","YearBuilt"]]# 特征
y = train.loc[:1000, "MSSubClass"] # 目标变量
print(X)
# 使用train_test_split划分数据集
X_train, y_train = SMOTE().fit_resample(X, y)
X_test,y_test= X,y
# 创建 Logistic Regression 模型
logreg = LogisticRegression(max_iter=1000)
# 训练模型
logreg.fit(X_train, y_train)
# 在测试集上进行预测
y_pred = logreg.predict(X_test)
# 输出结果
print("Predicted values:", y_pred)
predictions = pd.DataFrame({'COL_A': y_pred})
# 保存为 CSV 文件
predictions.to_csv('OUTPUT.csv', index=False)
# 绘制逻辑回归结果
precision, recall, thresholds = precision_recall_curve(y_test, logreg.decision_function(X_test))
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.')
plt.xlim([0.0, 1.0])
plt.ylim([0.0,1.0])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()
