#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Import the libraries
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


# Import .csv file and convert it to a DataFrame object
df = pd.read_csv("Wholesale customers data.csv");

print(df.head())


# In[ ]:


print(df.info())


# In[ ]:


df.drop(['Channel','Region'],axis=1,inplace=True)


# In[ ]:


array=df.values


# In[ ]:


array


# In[ ]:


stscaler = StandardScaler().fit(array)
X = stscaler.transform(array)


# In[ ]:


X


# In[ ]:


dbscan = DBSCAN(eps=0.8, min_samples=6)
dbscan.fit(X)


# In[ ]:


#Noisy samples are given the label -1.
dbscan.labels_


# In[ ]:


cl=pd.DataFrame(dbscan.labels_,columns=['cluster'])


# In[ ]:


cl


# In[ ]:


pd.concat([df,cl],axis=1)

