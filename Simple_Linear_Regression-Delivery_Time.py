#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import seaborn as sns


# In[3]:


del_time = pd.read_csv('delivery_time.csv')


# In[4]:


del_time


# In[5]:


del_time.info()


# # Correlation

# In[6]:


del_time.corr()


# In[10]:


sns.distplot(del_time['Delivery Time'])


# In[9]:


sns.distplot(del_time['Sorting Time'])


# In[11]:


del_time.columns = ['Delivery' , 'Sorting']
del_time


# In[12]:


model = smf.ols("Delivery ~ Sorting" , data = del_time).fit()


# In[13]:


sns.regplot(x = 'Sorting' , y = 'Delivery' , data = del_time)


# In[14]:


model.params


# In[15]:


model.tvalues , model.pvalues


# In[16]:


model.rsquared


# # Predict Values

# In[17]:


newdata = pd.Series([2 , 4 , 6 , 8 , 10])


# In[18]:


data_pred=pd.DataFrame(newdata,columns=['Sorting'])


# In[19]:


model.predict(data_pred)


# In[ ]:




