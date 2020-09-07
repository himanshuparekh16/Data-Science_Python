#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy import stats
import pandas as pd
import numpy as np


# In[4]:


glaxo_df=pd.read_csv('glaxo.csv')


# In[13]:


glaxo_df_ci = stats.norm.interval(0.95,
loc = glaxo_df.mean(),
scale = glaxo_df.std())
print( 'Gain at 95% confidence interval is:', np.round(glaxo_df_ci, 4))


# In[14]:


beml_df=pd.read_csv('beml.csv')
beml_df_ci = stats.norm.interval(0.95,
loc=beml_df.mean(),
scale=beml_df.std())


# In[15]:


from scipy import stats
stats.norm.ppf(.975)


# In[16]:


stats.norm.interval(0.95,0,1)

