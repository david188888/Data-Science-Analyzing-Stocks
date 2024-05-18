#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# In[3]:


data = pd.read_csv('data/apple.csv')
colume_drop = data.iloc[:, -2:]
data = data.drop(colume_drop, axis=1)
data['Date'] = pd.to_datetime(data['Date'], utc=True)
data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')
print(data.head())
print(type(data.index[0]))


# # 数据预处理

# In[33]:


for column in data.columns:
    if data[column].isnull().sum() > 0:
        print(column)
    else:
        print(column, '无缺失值')
        
        

# 数据归一化

scaler = MinMaxScaler(feature_range=(-1, 1))
close_data = data['Close'].values.reshape(-1, 1)
# print(close_data[:5])

prices_scaled  = scaler.fit_transform(close_data)


# 划分训练集和测试集
train_size = int(len(prices_scaled) * 0.7)
test_size = len(prices_scaled) - train_size
train_data, test_data = prices_scaled[0:train_size], prices_scaled[train_size:len(prices_scaled)]

# 构建数据集
def create_dataset(dataset, time_step):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 60
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# 转换数据维度
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))




# In[ ]:





# In[34]:


## 构建LSTM模型
from keras.models import Sequential
from keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
print(model.summary())



# In[35]:


# 训练模型
model.fit(X_train, y_train, epochs=100, batch_size=32)


# In[36]:


# 预测股票价格
import matplotlib.pyplot as plt
predicted_prices = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

test_data = scaler.inverse_transform(test_data)
# 可视化预测结果
fig = plt.figure(figsize=(12, 6))
plt.plot(data.index[train_size + time_step + 1:], test_data[time_step + 1:], color='blue', label='Actual Price')
plt.plot(data.index[train_size + time_step + 1:], predicted_prices, color='red', label='Predicted Price')
plt.title('Stock Price Prediction using LSTM')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()



# In[38]:


# 假设 time_step 已定义
# 假设 scaler 和 model 是你已经训练好的 MinMaxScaler 和 LSTM 模型

# 获取最后 time_step 天的数据作为输入
last_sequence = X_test[-1:]
current_sequence = last_sequence
# print(current_sequence)

# 存储模型预测的未来一周的价格
future_prices = []

# 连续预测未来一周
for _ in range(30):  # 未来7天
    next_day_price = model.predict(current_sequence)
    future_prices.append(next_day_price[0, 0])  # 存储预测结果
    # 更新输入数据用于下一次预测
    current_sequence = np.roll(current_sequence, -1, axis=1)
    current_sequence[0, -1, 0] = next_day_price[0, 0]

# 反归一化预测结果
future_prices = np.array(future_prices).reshape(-1, 1)
future_prices = scaler.inverse_transform(future_prices)

# 计算未来日期
last_date = data.index[-1]  # 假设数据集中有日期索引
future_dates = pd.date_range(start=last_date, periods=30)  # 生成未来7天的日期

# 可视化结果
fig = plt.figure(figsize=(12, 6))
plt.plot(data.index[train_size + time_step + 1:], test_data[time_step + 1:], color='blue', label='Actual Price')
plt.plot(data.index[train_size + time_step + 1:], predicted_prices, color='red', label='Predicted Price')
plt.plot(future_dates, future_prices, color='green', label='Future Predicted Price')  # 未来一周的预测
plt.title('Stock Price Prediction using LSTM')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

