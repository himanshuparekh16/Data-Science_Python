#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing 


# In[ ]:


Walmart = pd.read_csv("footfalls.csv")
Walmart.Footfalls.plot()


# # Splitting data

# In[ ]:


Train = Walmart.head(147)
Test = Walmart.tail(12)


# # Moving Average 

# In[ ]:


plt.figure(figsize=(12,4))
Walmart.Footfalls.plot(label="org")
for i in range(2,24,6):
    Walmart["Footfalls"].rolling(i).mean().plot(label=str(i))
plt.legend(loc='best')


# # Time series decomposition plot 
# 

# In[ ]:


decompose_ts_add = seasonal_decompose(Walmart.Footfalls,period=12)
decompose_ts_add.plot()
plt.show()


# # ACF plots and PACF plots
# 

# In[ ]:


import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(Walmart.Footfalls,lags=12)
tsa_plots.plot_pacf(Walmart.Footfalls,lags=12)
plt.show()


# ### Evaluation Metric MAPE

# In[ ]:


def MAPE(pred,org):
    temp = np.abs((pred-org)/org)*100
    return np.mean(temp)


# ### Simple Exponential Method
# 

# In[ ]:


ses_model = SimpleExpSmoothing(Train["Footfalls"]).fit(smoothing_level=0.2)
pred_ses = ses_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_ses,Test.Footfalls) 


# ### Holt method 

# In[ ]:


# Holt method 
hw_model = Holt(Train["Footfalls"]).fit(smoothing_level=0.8, smoothing_slope=0.2)
pred_hw = hw_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hw,Test.Footfalls) 


# ### Holts winter exponential smoothing with additive seasonality and additive trend
# 

# In[ ]:


hwe_model_add_add = ExponentialSmoothing(Train["Footfalls"],seasonal="add",trend="add",seasonal_periods=12).fit() #add the trend to the model
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_add_add,Test.Footfalls) 


# ### Holts winter exponential smoothing with multiplicative seasonality and additive trend

# In[ ]:


hwe_model_mul_add = ExponentialSmoothing(Train["Footfalls"],seasonal="mul",trend="add",seasonal_periods=12).fit() 
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_mul_add,Test.Footfalls)


# ## Final Model by combining train and test

# In[ ]:


hwe_model_add_add = ExponentialSmoothing(Walmart["Footfalls"],seasonal="add",trend="add",seasonal_periods=12).fit()


# In[ ]:


#Forecasting for next 10 time periods
hwe_model_add_add.forecast(10)


# In[ ]:


Walmart.tail()


# In[ ]:




