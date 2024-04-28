import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve

df=pd.read_csv('train.csv',usecols=[0,1,2,3,4,5])
count_null_series = df.isnull().sum() # returns series
count_null_df = pd.DataFrame(data=count_null_series, columns=['Num_Nulls'])
# what % of the null values take for that column
pct_null_df = pd.DataFrame(data=count_null_series/len(df), columns=['Pct_Nulls'])
null_stats = pd.concat([count_null_df, pct_null_df],axis=1)
print(null_stats)
print(df.head(11))
df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].mean()).round(1)
print(df.head(11))
