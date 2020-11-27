#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!pip install lifelines


# In[1]:


import pandas as pd
from lifelines import KaplanMeierFitter


# In[2]:


# Loading the the survival un-employment data
survival_unemp = pd.read_csv("survival_unemployment.csv")


# In[4]:


survival_unemp.head()
#survival_unemp.describe()


# In[5]:


survival_unemp["spell"].describe()


# In[6]:


# Spell is referring to time 
T = survival_unemp.spell


# In[7]:


# Initiating the KaplanMeierFitter model
kmf = KaplanMeierFitter()


# In[8]:


# Fitting KaplanMeierFitter model on Time and Events for death 
kmf.fit(T,event_observed=survival_unemp.event)
# Time-line estimations plot 
kmf.plot()


# In[9]:


# Over Multiple groups 
# For each group, here group is ui
survival_unemp.ui.value_counts()


# In[10]:


# Applying KaplanMeierFitter model on Time and Events for the group "1"
kmf.fit(T[survival_unemp.ui==1], survival_unemp.event[survival_unemp.ui==1], label='1')
ax = kmf.plot()

# Applying KaplanMeierFitter model on Time and Events for the group "0"
kmf.fit(T[survival_unemp.ui==0], survival_unemp.event[survival_unemp.ui==0], label='0')
kmf.plot(ax=ax)


# In[ ]:




