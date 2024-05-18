#!/usr/bin/env python
# coding: utf-8

# In[1]:


import yfinance as yf
import pandas as pd
import numpy as np


# ## 导入数据，检查数据类型，学习ticker的含义

# In[2]:


msft = yf.Ticker("MSFT")
print(msft)
msft.info


# In[4]:


msft.history(period="max")

