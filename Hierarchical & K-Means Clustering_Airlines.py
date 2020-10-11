#!/usr/bin/env python
# coding: utf-8

# # Hierarchical Clustering

# In[2]:


import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sn


# In[5]:


airlines = pd.read_csv('EastWestAirlines.csv')


# In[6]:


airlines


# In[7]:


# Normalization function 
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)


# In[9]:


# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(airlines.iloc[:,1:])


# In[10]:


# create dendrogram
dendrogram = sch.dendrogram(sch.linkage(df_norm, method='average'))


# In[11]:


# create clusters
hc = AgglomerativeClustering(n_clusters=4, affinity = 'euclidean', linkage = 'single')


# In[12]:


# save clusters for chart
y_hc = hc.fit_predict(df_norm)
Clusters=pd.DataFrame(y_hc,columns=['Clusters'])


# In[14]:


airlines['h_clusterid'] = hc.labels_


# In[15]:


airlines


# In[17]:


Clusters


# In[ ]:





# In[ ]:





# # K-Means

# In[18]:


from sklearn.cluster import KMeans


# In[22]:


# Normalization function 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_airlines_df = scaler.fit_transform(airlines.iloc[:,1:])


# In[23]:


# How to find optimum number of  cluster
#The K-means algorithm aims to choose centroids that minimise the inertia, or within-cluster sum-of-squares criterion:

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i,random_state=0)
    kmeans.fit(scaled_airlines_df)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[25]:


#Build Cluster algorithm
from sklearn.cluster import KMeans
clusters_new = KMeans(6, random_state=42)
clusters_new.fit(scaled_airlines_df)


# In[26]:


#Assign clusters to the data set
airlines['clusterid_new'] = clusters_new.labels_


# In[27]:


#these are standardized values.
clusters_new.cluster_centers_


# In[28]:


airlines


# In[ ]:




