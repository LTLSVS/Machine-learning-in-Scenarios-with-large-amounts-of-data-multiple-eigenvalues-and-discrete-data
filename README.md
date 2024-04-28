# Machine-learning-in-Scenarios-with-large-amounts-of-data-multiple-eigenvalues-and-discrete-data
LogisticRegression/ KNeighborsClassifier/ DecisionTreeClassifier
针对大数据量，多特征值，数据高度离散化的用户行为预测。结果为三种渠道A,B,C的二分类结果。
数据清洗：
1.先大体预览数据print(?.head())
主要目的是观察各特征值数据类型，大部分机器学习模型特征值的输入都是不支持'str'的。
对于特征值中'str'部分，def函数或利用lamda函数，将'str'映射为'int'或'float'
2.检查空值(check_0.py)和严重偏离值
