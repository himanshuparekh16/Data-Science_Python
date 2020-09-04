#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np


# In[9]:


from scipy import stats


# In[10]:


beml_df = pd.read_csv("BEML.csv")
beml_df[0:5]


# In[12]:


glaxo_df = pd.read_csv("GLAXO.csv")
glaxo_df[0:5]


# In[13]:


beml_df = beml_df[['Date', 'Close']]
glaxo_df = glaxo_df[['Date', 'Close']]


# In[14]:


beml_df


# In[15]:


'''The DataFrames have a date column, so we can
create a DatetimeIndex index from this column Date. It will ensure that the rows are sorted by time in
ascending order.'''
glaxo_df = glaxo_df.set_index(pd.DatetimeIndex(glaxo_df['Date']))
beml_df = beml_df.set_index(pd.DatetimeIndex(beml_df['Date']))


# In[16]:


glaxo_df


# In[17]:


import matplotlib.pyplot as plt
import seaborn as sn
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(glaxo_df.Close);
plt.xlabel('Time');
plt.ylabel('Close Price');


# In[18]:


plt.plot(beml_df.Close);
plt.xlabel('Time');
plt.ylabel('Close');


# In[19]:


glaxo_df['gain'] = glaxo_df.Close.pct_change(periods = 1)
beml_df['gain'] = beml_df.Close.pct_change(periods = 1)


# In[20]:


beml_df


# In[21]:


#drop first row since it is NaN
glaxo_df = glaxo_df.dropna()
beml_df = beml_df.dropna()


# In[23]:


#Plot the gains
plt.figure(figsize = (8, 6));
plt.plot(glaxo_df.index, glaxo_df.gain);
plt.xlabel('Time');
plt.ylabel('gain');


# In[24]:


sn.distplot(glaxo_df.gain, label = 'Glaxo');
plt.xlabel('gain');
plt.ylabel('Density');
plt.legend();


# In[25]:


sn.distplot(beml_df.gain, label = 'BEML');
plt.xlabel('gain');
plt.ylabel('Density');
plt.legend();


# In[26]:


print('Mean:', round(glaxo_df.gain.mean(), 4))
print('Standard Deviation: ', round(glaxo_df.gain.std(), 4))


# In[27]:


print('Mean: ', round(beml_df.gain.mean(), 4))
print('Standard Deviation: ', round(beml_df.gain.std(), 4))


# ### Probability of making 2% loss or higher in Glaxo

# In[34]:


stats.norm.cdf( -0.02,
loc=glaxo_df.gain.mean(),
scale=glaxo_df.gain.std())


# ### Probability of making 2% gain or higher in Glaxo

# In[35]:


1 - stats.norm.cdf(0.02,
loc=glaxo_df.gain.mean(),
scale=glaxo_df.gain.std())


# ### Probability of making 2% loss or higher in BEML

# In[36]:


stats.norm.cdf(-0.02,
loc = beml_df.gain.mean(),
scale = beml_df.gain.std())


# ### Probability of making 2% gain or higher in BEML

# In[37]:


1 - stats.norm.cdf(0.02,
loc = beml_df.gain.mean(),
scale = beml_df.gain.std())

