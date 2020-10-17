#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder


# In[13]:


groceries = pd.read_csv('groceries.csv')
groceries.head()


# In[14]:


df=pd.get_dummies(groceries)
df.head()


# In[30]:


frequent_itemsets = apriori(df, min_support=0.015, use_colnames=True)
frequent_itemsets


# In[35]:


rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.05)
rules


# In[36]:


rules.sort_values('lift',ascending = False)


# In[39]:


rules[rules.lift>1]


# In[ ]:





# In[ ]:




