import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve
from imblearn.over_sampling import SMOTE
from collections import Counter

df = pd.DataFrame()
df = pd.read_csv('/work/data/train_base_data.csv',usecols=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,18,19,20,21])
df_test= pd.read_csv('/work/data/test_data.csv',usecols=[1,2,3,4,5,6,7,8,9,10,11,15,16,17,18])
df = df.dropna(axis=0, how='any')
df_test = df_test.dropna(axis=0, how='any')
X = df.loc[:, ["COL1","COL2","COL3","COL4","COL5","COL6","COL7","COL8","COL9","COL10","COL11","COL15","COL16","COL17"]]#
sm = SMOTE(random_state=42)

X_resampled, y_resampled = sm.fit_resample(X, y)

X_train = X_resampled
y_train = y_resampled

print(X_train)