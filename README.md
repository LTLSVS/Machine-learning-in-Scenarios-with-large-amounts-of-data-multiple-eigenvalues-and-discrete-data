# Machine-learning-in-Scenarios-with-large-amounts-of-data-multiple-eigenvalues-and-discrete-data
LogisticRegression/ KNeighborsClassifier/ DecisionTreeClassifier
'''招商银行2024Fintech 数据赛道初赛题'''
'''本项目无竞赛相关数据，只提供代码参考，可用本地数据测试代码可行性'''
针对大数据量，多特征值，数据高度离散化的用户行为预测。结果为三种渠道A,B,C的二分类结果。
数据清洗：
1.先大体预览数据print(?.head())
主要目的是观察各特征值数据类型，大部分机器学习模型特征值的输入都是不支持'str'的。
对于特征值中'str'部分，def函数或利用lamda函数，将'str'映射为'int'或'float'
2.检查空值比例(check_0.py)和严重偏离值
删除空值比例过高的列（特征）。
空值比例较低的列，选择0填充，均值填充或中位数填充（依数据的分布情况决定）。
若行（属性）中存在严重偏离值，删除行。
应用模型：
