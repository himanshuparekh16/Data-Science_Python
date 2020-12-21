#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn import preprocessing


# In[2]:


data=pd.read_csv("book.csv")


# In[3]:


data.head()


# In[4]:


da=[]
for i in range(1,100,+1):
    i=i/100
    frequent_itemsets = apriori(data, min_support= i,use_colnames=True)
    da.append(frequent_itemsets.shape[0])
da=pd.DataFrame(da)
da=da.reset_index()
da=da.rename({0:'Number_of_Elements','index':'Support_percentage'},axis=1)
sns.scatterplot(data=da,x='Support_percentage',y='Number_of_Elements')


# In[5]:



frequent_itemsets = apriori(data, min_support= 0.10,use_colnames=True)
frequent_itemsets


# In[6]:


dat=[]
for i in range(1,400,1):
    i=i/100
    rules1= association_rules(frequent_itemsets, metric='lift', min_threshold=i)
    dat.append(rules1.shape[0])
dat=pd.DataFrame(dat)
dat=dat.reset_index()
dat=dat.rename({'index':'Lift_ratio/100',0:'Number_of_elements'},axis=1)
sns.scatterplot(data=dat,x='Lift_ratio/100',y='Number_of_elements')
plt.title('Lift ratio vs Number of Elements at specified Support')
plt.show()


# In[7]:


dat_C=[]
for i in range(1,100,1):
    i=i/100
    rules2= association_rules(frequent_itemsets, metric='confidence', min_threshold=i)
    dat_C.append(rules2.shape[0])
dat_C=pd.DataFrame(dat_C)
dat_C=dat_C.reset_index()
dat_C=dat_C.rename({'index':'Confidence_Percentage',0:'Number_of_elements'},axis=1)
sns.scatterplot(data=dat_C,x='Confidence_Percentage',y='Number_of_elements')
plt.title('Confidence ratio vs Number of Elements at specified Support')
plt.show()


# In[8]:


rules= association_rules(frequent_itemsets, metric='lift', min_threshold=2)
rules=rules.reset_index()
rules


# In[9]:


sns.scatterplot(data=rules,x='index',y='lift')
plt.show()


# In[10]:


sns.scatterplot(data=rules,x='index',y='confidence')
plt.show()


# In[ ]:




