#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import seaborn as sns


# In[2]:


salary = pd.read_csv('salary_data.csv')


# In[3]:


salary


# In[4]:


salary.info()


# In[6]:


sns.distplot(salary['YearsExperience'])


# In[7]:


sns.distplot(salary['Salary'])


# In[8]:


sns.boxplot(salary['YearsExperience'])


# In[9]:


sns.boxplot(salary['Salary'])


# In[10]:


salary[salary.duplicated()]


# In[11]:


salary.hist()


# In[12]:


salary.corr()


# In[13]:


import statsmodels.formula.api as smf


# In[14]:


salary_model = smf.ols('Salary ~ YearsExperience' , data = salary).fit()


# In[15]:


sns.regplot(x = 'YearsExperience' , y = 'Salary' , data = salary)


# In[16]:


salary_model.params


# In[17]:


#t and p-Values
salary_model.tvalues, salary_model.pvalues


# In[18]:


salary_model.rsquared


# In[19]:


sal_data = pd.Series([2.5 , 5 , 7.5 , 10])


# In[20]:


sal_pred = pd.DataFrame(sal_data , columns=['YearsExperience'])


# In[21]:


sal_pred


# In[22]:


salary_model.predict(sal_pred)


# In[ ]:




