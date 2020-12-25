#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder


# In[2]:


data=pd.read_csv('my_movies.csv')
data.head()


# In[3]:


data1=data.iloc[:,5:].copy()
#data1=data1.astype(bool)
def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

data1 = data1.applymap(encode_units)
data1


# In[4]:


da=[]
for i in range(1,100,+1):
    i=i/100
    frequent_itemsets = apriori(data1, min_support= i,use_colnames=True)
    da.append(frequent_itemsets.shape[0])
da=pd.DataFrame(da)
da=da.reset_index()
da=da.rename({0:'Number_of_Elements','index':'Support_percentage'},axis=1)
sns.scatterplot(data=da,x='Support_percentage',y='Number_of_Elements')


# In[5]:


frequent_items= apriori(data1,min_support=0.2, use_colnames=True)
frequent_items


# In[8]:


rules=association_rules(frequent_items, metric='lift',min_threshold=0.7)
rules=rules.reset_index()


# In[9]:


rules.sort_values('lift',ascending = False)[0:20]


# In[10]:


sns.scatterplot(data=rules,x='index',y='lift')


# In[11]:


sns.scatterplot(data=rules,x='index',y='confidence')


# In[ ]:




