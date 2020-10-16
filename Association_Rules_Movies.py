#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder


# In[5]:


movies = pd.read_csv('my_movies.csv')
movies


# In[8]:


df=pd.get_dummies(movies)
df.head()


# In[15]:


frequent_itemsets1 = apriori(df, min_support=0.5, use_colnames=True)
frequent_itemsets1


# In[21]:


rules = association_rules(frequent_itemsets1, metric="lift", min_threshold=1.0)
rules


# In[22]:


rules.sort_values('lift',ascending = False)


# In[23]:


rules[rules.lift>1]


# In[ ]:




