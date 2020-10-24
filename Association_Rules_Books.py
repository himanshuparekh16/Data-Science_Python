#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder


# In[24]:


book = pd.read_csv("book.csv")
book.head()


# In[25]:


df=pd.get_dummies(book)
df.head()


# In[26]:


frequent_itemsets = apriori(df, min_support=0.15, use_colnames=True)
frequent_itemsets


# In[27]:


rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.07)
rules


# In[28]:


rules.sort_values('lift',ascending = False)[0:10]


# In[29]:


rules[rules.lift>1]


# In[ ]:





# In[ ]:




