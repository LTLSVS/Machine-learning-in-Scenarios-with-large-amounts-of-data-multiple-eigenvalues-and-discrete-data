import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency


def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

def create_cramers_v_matrix(data):
    """
    Create Cramer's V correlation matrix for given dataframe
    """
    num_columns = len(data.columns)
    correlation_matrix = pd.DataFrame(np.zeros((num_columns, num_columns)), columns=data.columns, index=data.columns)

    # 使用Cramer's V填充相关系数矩阵
    for i in data.columns:
        for j in data.columns:
            correlation_matrix.loc[i, j] = cramers_v(data[i], data[j])

    return correlation_matrix


def A_N1(x):
    if x["COL3"] == 'A':
        return 1
    elif x["COL3"] == 'B':
        return 2
    elif x["COL3"] == 'C':
        return 3
    else:
        return 0
def A_N2(x):
    if x["COL4"] == 'A':
        return 1
    elif x["COL4"] == 'B':
        return 2
    elif x["COL4"] == 'C':
        return 3
    else:
        return 0
def A_N3(x):
    if x["COL5"] == 'A':
        return 1
    elif x["COL5"] == 'B':
        return 2
    elif x["COL5"] == 'C':
        return 3
    else:
        return 0
def C19(x):
    if x["COL19"] == 'D':
        return 1
    elif x["COL19"] == 'E':
        return 2
    elif x["COL19"] == 'F':
        return 3
    else:
        return 0

df = pd.DataFrame()
train = pd.read_csv('/work/data/train_base_data.csv',usecols=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,18,19,20,21])


df["CHANNEL_A"] = train.loc[:,"CHANNEL_A"]
df["CHANNEL_B"] = train.loc[:,"CHANNEL_B"]
df["CHANNEL_C"] = train.loc[:,"CHANNEL_C"]
df["A"] = train.iloc[:50000,0]
df["B"] = train.iloc[:50000,1]
df["C"] = train.iloc[:50000,2]
df["COL3"] = train.loc[:50000,"COL3"]
df["COL4"] = train.loc[:50000,"COL4"]
df["COL5"] = train.loc[:50000,"COL5"]
df["COL3"] = df.apply(A_N1,axis=1)
df["COL4"] = df.apply(A_N2,axis=1)
df["COL5"] = df.apply(A_N3,axis=1)
df["COL6"] = train.loc[:50000,"COL6"]
df["COL7"] = train.loc[:50000,"COL7"]
df["COL8"] = train.loc[:50000,"COL8"]
df["COL9"] = train.loc[:50000,"COL9"]
df["COL15"] = train.loc[:50000,"COL15"]
df["COL16"] = train.loc[:50000,"COL16"]
df["COL17"] = train.loc[:50000,"COL17"]
df["COL18"] = train.loc[:50000,"COL18"]
df["COL19"] = train.loc[:50000,"COL19"]
df["COL22"] = train.loc[:50000,"COL22"]
df["COL23"] = train.loc[:50000,"COL23"]
df["COL24"] = train.loc[:50000,"COL24"]
df["COL25"] = train.loc[:50000,"COL25"]
df["COL36"] = train.loc[:50000,"COL36"]
df["COL37"] = train.loc[:50000,"COL37"]
df["COL38"] = train.loc[:50000,"COL38"]
df["COL15"] = df.apply(lambda x: 3 * x["COL18"] - 3 * x["COL15"], axis=1)
df["COL19"] = df.apply(C19,axis=1)


print(df)

correlation_matrix = create_cramers_v_matrix(df)
print(correlation_matrix)
