#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[5]:


flights = pd.read_csv("flights.csv")


# # Statistical Analysis

# In[10]:


flights.head()


# In[11]:


flights.tail(10)


# In[16]:


flights.columns


# In[17]:


flights.count()


# In[19]:


len(flights.columns)


# In[20]:


flights.describe()


# In[24]:


flights[['SCHED_DEP']].max()


# In[25]:


flights.DEP_DELAY.min()


# In[26]:


flights.mean()


# In[27]:


flights.MONTH.median()


# In[28]:


flights.std()


# In[29]:


flights.DIST.std()


# In[32]:


flights.sample()


# In[34]:


flights.shape


# In[35]:


flights.head(20).describe()


# In[36]:


flights.AIR_TIME


# In[38]:


flights[['SCHED_ARR']]


# # Groupby

# In[43]:


flights.groupby(['MONTH']).mean()


# In[51]:


flights.groupby(['DIVERTED'])[['AIRLINE','AIR_TIME','CANCELLED']].describe()


# In[54]:


flights.DIST > 500


# In[57]:


flights[flights['DIST']>700]


# # iloc

# In[59]:


flights[flights.AIRLINE=='AA']


# In[62]:


flights.iloc[0:10 , 2:5]


# In[63]:


flights.iloc[3,]


# In[66]:


flights.iloc[: , -4]

