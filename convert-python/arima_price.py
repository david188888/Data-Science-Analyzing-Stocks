#!/usr/bin/env python
# coding: utf-8

# In[182]:


import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.ticker import ScalarFormatter


# ## 1. 数据预处理

# In[183]:


apple = yf.Ticker("AAPL")
data = apple.history(start='2019-01-01', end='2024-05-06')
# print(type(data))

# print(data.head())
# print(data.columns)
# print(data.index)
# 进行数据预处理

def write_to_csv(data, filename):
    if not os.path.exists('data'):
        os.makedirs('data')
    data.to_csv('data/' + filename)
    
write_to_csv(data, 'apple.csv')
        
colume_drop = data.iloc[:, -2:]
# print(colume_drop.head())
data = data.drop(colume_drop, axis=1)
print(data.head())
data = data.fillna(method='ffill')
for column in data.columns:
    if data[column].isnull().sum() > 0:
        print(column)
    else:
        print(column, '无缺失值')




# ## 初步对数据进行可视化分析，查看数据的分布情况，以及数据之间的相关性

# In[184]:


# 画出股票的走势图
fig, ax = plt.subplots(2, 2, figsize=(15, 10))
ax[0,0].plot(data['Open'])
ax[0,0].set_title('Open')
ax[0,1].plot(data['High'])
ax[0,1].set_title('High')
ax[1,0].plot(data['Close'])
ax[1,0].set_title('Close')
ax[1,1].plot(data['Low'])
ax[1,1].set_title('Low')
plt.show()


# In[185]:


# 画出股票的成交量
plt.figure(figsize=(15, 10))
plt.plot(data['Volume'])
ax = plt.gca()
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
plt.title('Volume')
plt.show()


# ## 使用ARIMA模型进行预测

# In[186]:


from statsmodels.tsa.stattools import adfuller

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, plot_predict
close_price = data['Close']
open_price = data['Open']
volume = data['Volume']

# 进行ADF检验
result = adfuller(close_price)
ADF_statistic = result[0]
p_value = result[1]
critical_values = result[4]
print('ADF Statistic:', ADF_statistic)
print('p-value:', p_value)
print('Critical Values:', critical_values)

if p_value < 0.05:
    print('平稳')
    plot_acf(close_price)
    plot_pacf(close_price)
else:
    print('不平稳')
    diff_close_price = close_price.diff().dropna()
    diff_result = adfuller(diff_close_price)
    diff_ADF_statistic = diff_result[0]
    diff_p_value = diff_result[1]
    print('ADF Statistic:', diff_ADF_statistic)
    print('p-value:', diff_p_value)
    plot_acf(diff_close_price)
    # 绘制ACF和PACF图
    plot_pacf(diff_close_price)
    







# ## 进行ARIMA模型预测

# In[187]:


from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm



# 将数据分为训练集和测试集
train = close_price[:'2023-01-01']
test = close_price['2023-01-01':]




# ## 进行模型评估

# In[191]:


# 进行预测
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=(1,1,1)).fit()
    output = model.forecast()
    output = output.tolist()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    # print('predicted=%f, expected=%f' % (yhat, obs))
    
    
# 计算RMSE
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)

    
# 画出预测结果
plt.figure(figsize=(15, 10))
predict_result = pd.Series(predictions, index=test.index)
plt.plot(close_price)
plt.plot(predict_result, color='red')


# In[200]:


model = ARIMA(close_price, order=(1,1,1)).fit()


# 进行AIC 和 BIC的评估
print('AIC:', model.aic)
print('BIC:', model.bic)
# 残差分析
residuals = model.resid

# fig = plt.figure(figsize=(15, 10))
# plt.plot(residuals)
# plt.title('Residuals')
# plt.show()

# # 画出残差的密度图
# fig = plt.figure(figsize=(15, 10))
# sns.kdeplot(residuals)
# plt.title('Residuals Density')
# plt.show()

# residual_pd = pd.DataFrame(residuals)
# print(residual_pd.describe())

future_price = model.forecast(steps=100)
future_price = future_price.tolist()
print(future_price)
future_index = pd.date_range('2024-05-07', periods=100)
future_price = pd.Series(future_price, index=future_index)
plt.figure(figsize=(15, 10))
plt.plot(close_price)
plt.plot(future_price, color='red')
plt.show()

