#!/usr/bin/env python
# coding: utf-8

# # Import Data Set

# In[1]:


import pandas as pd
data = pd.read_csv("NewspaperData.csv")


# In[2]:


data


# In[3]:


data.info()


# # Correlation

# In[4]:


data.corr()


# In[5]:


import seaborn as sns
sns.distplot(data['daily'])


# In[6]:


import seaborn as sns
sns.distplot(data['sunday'])


# Fitting a Linear Regression Model

# In[7]:


import statsmodels.formula.api as smf
model = smf.ols("sunday~daily",data = data).fit()


# In[8]:


sns.regplot(x="daily", y="sunday", data=data);


# In[9]:


#Coefficients
model.params


# In[10]:


#t and p-Values
print(model.tvalues, '\n', model.pvalues)    


# In[11]:


#R squared values
(model.rsquared,model.rsquared_adj)


# # Predict for new data point

# In[12]:


#Predict for 200 and 300 daily circulation
newdata=pd.Series([200,300])


# In[13]:


newdata


# In[14]:


data_pred=pd.DataFrame(newdata,columns=['daily'])


# In[15]:


data_pred


# In[18]:


model.predict(data_pred)


# In[ ]:




