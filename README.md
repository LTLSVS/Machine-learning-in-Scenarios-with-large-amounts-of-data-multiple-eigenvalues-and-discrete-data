# Machine-learning-in-Scenarios-with-large-amounts-of-data-multiple-eigenvalues-and-discrete-data
LogisticRegression/ KNeighborsClassifier/ DecisionTreeClassifier  
'''招商银行2024Fintech 数据赛道初赛题'''  
'''本项目无竞赛相关数据，只提供代码参考，可用本地数据测试代码可行性'''  
针对大数据量，多特征值，数据高度离散化的用户行为预测。结果为三种渠道A,B,C的二分类结果。  
评分方式为：精确率和召回率的综合值  
  
##数据清洗：  
1.先大体预览数据print(?.head())  
主要目的是观察各特征值数据类型，大部分机器学习模型特征值的输入都是不支持'str'的。  
对于特征值中'str'部分，def函数或利用lamda函数，将'str'映射为'int'或'float'  
2.检查空值比例(check_0.py)和严重偏离值  
删除空值比例过高的列（特征）。 
空值比例较低的列，选择0填充，均值填充或中位数填充（依数据的分布情况决定）。  
若行（属性）中存在严重偏离值，删除行。  
3.筛选特征(connection.py)，本题数据特征数量大，选择过多的特征会导致噪声过多，使模型过拟合，降低召回率。  
构建不同特征间的相关系数矩阵，因本题的数据高度离散化，故选择'Cramers_v'相关系数替代常用的'Person'相关系数，后者更加适用于线性相关特征。  
可视化相关系数矩阵。或采用热力图获取更直观结果。  
##应用模型(pred_A.py, pred_B.py, pred_C.py, SVM_A.py, combine_result.py)   
1.通过数据预览可知:B列的0，1样本比例大约为3:1，较为均衡。从B列入手尝试模型构建。从imblearn库import SMOTE，对B列进行过采样平衡数据。  
对二分类模型，先尝试LogisticRegression(逻辑回归)。B列使用LogisticRegression结果符合预期。  
2.再对A列套用B列模型，改用与A列相关度高的特征。模型结果较差，分析原因，A列数据严重不平衡，1样本数:0样本数 = 0.016，使用SMOTE时构建过多1样本，过拟合严重，鲁棒性极差，模型召回率低，且不适用训练数据以外的数据。  
改用ADASYN或Borderline SMOTE进行采样，过拟合问题有所改善，但由于原数据不平衡严重，模型准确率不足。增加KNeighbors(KNN)和DecisionTree(决策树)模型，使用Voting(投票法)综合三种模型结果得到最终结果。  
若想进一步优化，可在算力允许的情况下尝试使用SVM(支持向量机)模型，SVM对多特征值，数据高度离散化的数据样本效果更佳。  
3.C列同A列。  
综合数据，得出预测结果。  

