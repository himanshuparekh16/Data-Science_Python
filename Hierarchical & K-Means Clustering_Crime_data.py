#!/usr/bin/env python
# coding: utf-8

# # Hierarchical Clustering

# In[1]:


import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sn


# In[2]:


crime_data = pd.read_csv('crime_data.csv')


# In[3]:


crime_data


# In[4]:


# Normalization function 
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)


# In[6]:


# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(crime_data.iloc[:,1:])


# In[7]:


# create dendrogram
dendrogram = sch.dendrogram(sch.linkage(df_norm, method='average'))


# In[8]:


# create clusters
hc = AgglomerativeClustering(n_clusters=4, affinity = 'euclidean', linkage = 'single')


# In[9]:


# save clusters for chart
y_hc = hc.fit_predict(df_norm)
Clusters=pd.DataFrame(y_hc,columns=['Clusters'])


# In[11]:


crime_data['h_clusterid'] = hc.labels_


# In[12]:


crime_data


# In[13]:


Clusters


# In[ ]:





# In[ ]:





# # K-Means Clustering

# In[15]:


from sklearn.cluster import KMeans


# In[17]:


# Normalization function 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_airlines_df = scaler.fit_transform(crime_data.iloc[:,1:])


# In[18]:


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


# In[19]:


#Build Cluster algorithm
from sklearn.cluster import KMeans
clusters_new = KMeans(5, random_state=42)
clusters_new.fit(scaled_airlines_df)


# In[20]:


#Assign clusters to the data set
crime_data['clusterid_new'] = clusters_new.labels_


# In[21]:


#these are standardized values.
clusters_new.cluster_centers_


# In[22]:


crime_data


# In[ ]:




