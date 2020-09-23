#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import hierarchical clustering libraries
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sn


# In[2]:


Univ = pd.read_csv("Universities.csv")


# In[4]:


# Normalization function 
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)


# In[5]:


# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(Univ.iloc[:,1:])


# In[11]:


# create dendrogram
dendrogram = sch.dendrogram(sch.linkage(df_norm, method='average'))


# In[12]:


# create clusters
hc = AgglomerativeClustering(n_clusters=4, affinity = 'euclidean', linkage = 'single')


# In[13]:


# save clusters for chart
y_hc = hc.fit_predict(df_norm)
Clusters=pd.DataFrame(y_hc,columns=['Clusters'])


# In[15]:


Univ['h_clusterid'] = hc.labels_


# In[16]:


Univ


# In[14]:


Clusters


# In[ ]:




