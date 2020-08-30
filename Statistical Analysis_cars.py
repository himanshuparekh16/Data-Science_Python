#!/usr/bin/env python
# coding: utf-8

# In[31]:


import numpy as np
import pandas as pd
from statistics import mode


# In[3]:


cars = pd.read_csv("Q7.csv")


# In[29]:


cars.describe()


# In[7]:


cars.mean()


# In[40]:


cars.median()


# In[41]:


mode(cars['Score'])


# In[10]:


cars.std()


# In[11]:


cars.var()


# ### Range = max - min

# In[26]:


cars.max()


# In[27]:


cars.min()

