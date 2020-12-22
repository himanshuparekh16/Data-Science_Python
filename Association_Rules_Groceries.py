#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn import preprocessing
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder


# In[2]:


data=pd.read_csv("groceries.csv",header=None)


# In[3]:


data.head()


# In[4]:


data.dtypes


# In[5]:


data.fillna(0,inplace=True)


# In[6]:


def createlist(A_list,value):
    return list(filter(lambda x:x!=value,A_list))


# In[7]:


list_all_transaction=[]

for index,row in data.iterrows():
    transaction=row.values.tolist()
    transaction=createlist(transaction,0)
    list_all_transaction.append(transaction)
    
list_all_transaction


# In[8]:


da=TransactionEncoder()
da_bool=da.fit(list_all_transaction).transform(list_all_transaction)
data=pd.DataFrame(da_bool,columns=da.columns_)


# In[9]:


frequent_itemsets=apriori(data,min_support=0.03,use_colnames=True)
frequent_itemsets


# In[10]:


rules=association_rules(frequent_itemsets, metric="lift",min_threshold=1.5)
rules=rules.reset_index()


# In[11]:


rules.sort_values('lift',ascending = False)


# In[12]:


sns.scatterplot(data=rules,x='index',y='lift')


# In[13]:



sns.scatterplot(data=rules,x='index',y='confidence')


# In[ ]:




