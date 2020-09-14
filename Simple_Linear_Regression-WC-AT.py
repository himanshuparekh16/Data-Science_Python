#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns


# In[2]:


wc_at = pd.read_csv('WC_AT.csv')


# In[3]:


wc_at


# In[4]:


wc_at.info()


# # Correlation

# In[5]:


wc_at.corr()


# In[6]:


sns.distplot(wc_at['Waist'])


# In[7]:


sns.distplot(wc_at['AT'])


# In[8]:


import statsmodels.formula.api as smf


# In[9]:


model = smf.ols('AT~Waist' , data=wc_at).fit()


# In[10]:


sns.regplot(x = 'Waist' , y = 'AT' , data = wc_at)


# In[11]:


#Coefficients
model.params


# In[12]:


# t & p values
print(model.tvalues , '\n \n \n' , model.pvalues)


# In[13]:


# R-Squared values
(model.rsquared , model.rsquared_adj)


# In[ ]:





# # Predicted value for new data

# In[14]:


data = pd.Series([75 , 35 , 90])


# In[15]:


data


# In[16]:


data_pred = pd.DataFrame(data , columns=['Waist'])


# In[17]:


data_pred


# In[18]:


model.predict(data_pred)


# In[ ]:




