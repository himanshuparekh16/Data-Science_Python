#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot as plt
import seaborn as sn


# In[2]:


wine = pd.read_csv('wine.csv')


# In[3]:


wine.head()


# In[4]:


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale


# In[5]:


# Considering only numerical data 
wine.data = wine.iloc[:,0:]
wine.data.head(4)


# In[6]:


# Normalizing the numerical data 
wine_normal = scale(wine.data)


# In[7]:


pca = PCA(n_components = 14)
pca_values = pca.fit_transform(wine_normal)


# In[8]:


# The amount of variance that each PCA explains is 
var = pca.explained_variance_ratio_
var
pca.components_[0]


# In[9]:


# Cumulative variance 

var1 = np.cumsum(np.round(var,decimals = 4)*100)
var1


# In[10]:


# Variance plot for PCA components obtained 
plt.plot(var1,color="red")


# In[11]:


# plot between PCA1 and PCA2 
x = pca_values[:,0]
y = pca_values[:,1]
z = pca_values[:2:3]


# In[12]:


plt.scatter(x,y)


# ### Clustering

# In[13]:


new_df = pd.DataFrame(pca_values[:,0:4])


# In[14]:


from sklearn.cluster import KMeans


# In[15]:


kmeans = KMeans(n_clusters = 3)
kmeans.fit(new_df)
kmeans.labels_


# ### Hierarchical Clustering

# In[16]:


# Normalization function 
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)


# In[17]:


df_norm = norm_func(wine.iloc[:,1:])


# In[18]:


dendrogram = sch.dendrogram(sch.linkage(df_norm, method='average'))


# In[19]:


# create clusters
hc = AgglomerativeClustering(n_clusters=4, affinity = 'euclidean', linkage = 'single')


# In[20]:


# save clusters for chart
y_hc = hc.fit_predict(df_norm)
Clusters=pd.DataFrame(y_hc,columns=['Clusters'])


# In[21]:


wine['h_clusterid'] = hc.labels_


# In[22]:


wine


# In[25]:


Clusters


# In[ ]:




