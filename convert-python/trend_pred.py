#!/usr/bin/env python
# coding: utf-8

# In[75]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## 数据预处理

# In[76]:


df = pd.read_csv('data/apple.csv')
colume_drop = df.iloc[:, -2:]
df = df.drop(colume_drop, axis=1)
df = df.fillna(method='ffill')
for column in df.columns:
    if df[column].isnull().sum() > 0:
        print(column)
    else:
        print(column, '无缺失值')
df['Date'] = pd.to_datetime(df['Date'], utc= True)
df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')
print(df)


# ## 计算简单移动平均SMA 和指数移动平均EMA

# In[77]:


close_data = df['Close']
sma_close_short = close_data.rolling(window=20).mean()
sma_close_long = close_data.rolling(window=100).mean()
ema_close_short = close_data.ewm(span=20, adjust=False).mean()
ema_close_long = close_data.ewm(span=100, adjust=False).mean()



# 可视化
plt.figure(figsize=(20,10))
plt.plot(close_data.index, close_data, label='AAPL_close_p', color='blue')
plt.plot(sma_close_short.index, sma_close_short, label='20_days_sma', color='red')
plt.plot(sma_close_long.index, sma_close_long, label='100_days_sma', color='green')
plt.plot(ema_close_short.index, ema_close_short, label='20_days_ema', color='yellow')
plt.plot(ema_close_long.index, ema_close_long, label='100_days_ema', color='purple')
plt.xlabel('Date')
plt.autoscale(tight=True)
plt.title('AAPL_close_price and Moving Averages')
plt.legend()
# plt.show()


# ## 使用趋势线结合模型进行预测

# In[78]:


# 使用简单的logistic回归模型
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


target = (close_data.shift(-1) > close_data).astype(int)

features = pd.DataFrame(
    {
        'close': close_data,
        'sma_short': sma_close_short,
        'sma_long': sma_close_long,
    }
).dropna()
# print(features)

# 分离标签
target = target.reindex(features.index)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=False)

# 训练模型
model = LogisticRegression()
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)


# ## 使用SVR-tuned进行预测

# In[79]:


from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score



def model_SVR(x_train, y_train, validation_x):
    svr_model = SVR(kernel='linear')
    model = svr_model.fit(x_train, y_train)
    return model


# 使用SVM-TUNED模型
def model_SVRTuning(x_train, y_train, validation_x):
    hyperparameters_linearSVR = {
        'C':[0.5, 1.0, 10.0, 50.0, 100.0, 120.0,150.0, 300.0, 500.0,700.0,800.0, 1000.0],
        'epsilon':[0, 0.1, 0.5, 0.7, 0.9],
    }
    
    grid_search_SVR_feat = GridSearchCV(estimator=model_SVR(x_train, y_train, validation_x),
                           param_grid=hyperparameters_linearSVR,
                           cv=TimeSeriesSplit(n_splits=10),
    )

    model = grid_search_SVR_feat.fit(x_train, y_train)
    print('Best Parameters for SVR:', model.best_params_)
    
    return model


# 进行模型评估
def bestModel_validateResult(model, model_name, x_test,y_test):
    prediction = model.predict(x_test)
    prediction = pd.DataFrame(prediction, index=y_test.index)
    prediction_df = pd.concat([y_test, prediction], axis=1)
    prediction_df.columns = ['Actual', 'Prediction']
    print(prediction_df)
    RMSE_Score = np.sqrt(mean_squared_error(y_test, prediction))
    R2_Score = r2_score(y_test, prediction)
    plt.figure(figsize = (23,10))
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title(model_name + 'Prediction Vs Actual',fontsize = 20)
    plt.plot(y_test, label='test data',color = 'blue')
    plt.plot(prediction, label='prediction', color = 'red')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()
    print(model_name + ' RMSE: ', RMSE_Score)
    print(model_name + ' R2 score: ', R2_Score) 
    return RMSE_Score, R2_Score


# 进行数据归一化
from sklearn.preprocessing import StandardScaler

target_data = close_data.shift(-1).fillna(value=close_data.mean())
def normalize_split_data(data,target_data):
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    X_train, X_test, y_train, y_test = train_test_split(data, target_data, test_size=0.2, shuffle=False)
    return X_train, X_test, y_train, y_test

x_train, x_test, y_train, y_test = normalize_split_data(close_data.to_frame(), target_data.to_frame())

print(x_train)


# 使用SVM-TUNED模型
# svr_model = model_SVRTuning(x_train, y_train, x_test)
svr_model = SVR(kernel='linear', C=1000.0, epsilon=0)
svr_model.fit(x_train, y_train)
RMSE_Score, R2_Score = bestModel_validateResult(svr_model, 'SVR', x_test, y_test)




# In[80]:


# 预测未来一个月的股票趋势
def predict_future_trend(origin_data, data, model):
    prediction = model.predict(data)
    print(f"predicted price: {prediction}")
    
    # 可视化
    plt.figure(figsize=(20,10))
    plt.plot(origin_data.index, origin_data, label='AAPL_close_p', color='blue')
    plt.plot(data.index, prediction, label='Prediction', color='red')
    plt.xlabel('Date')
    plt.autoscale(tight=True)
    plt.title('AAPL_close_price and Prediction')
    plt.legend()
    plt.show()
    
    
input_data = close_data[-30:].to_frame()
scaler = StandardScaler()
input_data = scaler.fit_transform(input_data)
input_data = pd.DataFrame(input_data, index=close_data[-30:].index)
print(input_data)
predict_future_trend(close_data, input_data, svr_model)

