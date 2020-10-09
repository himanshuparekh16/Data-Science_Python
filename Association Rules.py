#!/usr/bin/env python
# coding: utf-8

# # Import libraries 

# In[2]:


get_ipython().system('pip install mlxtend')


# In[3]:


import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder
# conda install -c conda-forge mlxtend


# In[4]:


titanic = pd.read_csv("Titanic.csv")
titanic.head()


# # Pre-Processing
# As the data is not in transaction formation 
# We are using transaction Encoder

# In[5]:


df=pd.get_dummies(titanic)
df.head()


# # Apriori Algorithm 

# In[6]:


frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)
frequent_itemsets


# In[7]:


rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.7)
rules


# In[8]:


rules.sort_values('lift',ascending = False)[0:20]


# In[9]:


rules[rules.lift>1]


# In[ ]:




