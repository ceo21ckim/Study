import pandas as pd
import numpy as np

data = pd.DataFrame([
    [1, 7, 3, 5, 3, 5, 3, 5, 3, 5, 3, 5], 
    [1, 7, 1, 7, 1, 7, 1, 7, 1, 7, 1, 7],
    [1, 7, 1, 7, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
]).T

data.columns = ['스파이더맨', '괴물', '반도']

data
data_pre = data.fillna(data.loc[:,'반도'].mean())

cov_df = np.cov(data_pre, rowvar = False)
cov_df

np.cov(data, rowvar = 0)

# implementation covariance 
# 평균치 대체
covariance = []
for i in range(len(data.columns)):
    cov = []
    for j in range(len(data.columns)):
        x = data.iloc[:,i].copy()
        y = data.iloc[:,j].copy()
        
        x_mean = np.mean(x)
        x_std = np.std(x)
        
        y_mean = np.mean(y)
        y_std = np.std(y)
        
        cov.append(np.sum( (x - x_mean) * (y - y_mean)) / (len(x) - 1))
        
    covariance.append(cov)

np.array(covariance)


# 제거
covariance = []
for i in range(3):
    cov = []
    for j in range(3):
        xy = data.iloc[:,[i,j]].copy().dropna(axis = 0)
        
        x = xy.iloc[:,0].copy()
        y = xy.iloc[:,1].copy()
        
        x_mean = np.mean(x)
        x_std = np.std(x)
        
        y_mean = np.mean(y)
        y_std = np.std(y)
        
        cov.append(np.sum( (x - x_mean) * (y - y_mean)) / (len(x) - 1))

    
    covariance.append(cov)

np.array(covariance)


# 0으로 대체
data_t = data.fillna(0)

np.cov(data_t, rowvar = False)

covariance = []
for i in range(3):
    cov = []
    for j in range(3):
        x = data_t.iloc[:, i].copy()
        y = data_t.iloc[:, j].copy()
        
        x_mean = np.mean(x)
        x_std = np.std(x)
        
        y_mean = np.mean(y)
        y_std = np.std(y)
        
        cov.append(np.sum( (x - x_mean) * (y - y_mean)) / (len(x) - 1))

    
    covariance.append(cov)

np.array(covariance)