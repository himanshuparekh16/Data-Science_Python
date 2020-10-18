#!/usr/bin/env python
# coding: utf-8

# In[37]:


import pandas as pd 
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
import seaborn as sns


# In[6]:


wine = pd.read_csv('wine.csv')


# In[7]:


wine


# In[14]:


# Considering only numerical data 
wine_data = wine.iloc[:,0:]
wine_data.head()
# Converting into numpy array
wine_df = wine_data.values
wine_df


# In[15]:


# Normalizing the numerical data 
wine_normal = scale(wine_df)


# In[16]:


wine_normal


# In[19]:


pca = PCA()
pca_values = pca.fit_transform(wine_normal)


# In[20]:


pca_values


# In[22]:


pca = PCA(n_components = 14)
pca_values = pca.fit_transform(wine_normal)


# In[24]:


# The amount of variance that each PCA explains is 
var = pca.explained_variance_ratio_
var


# In[25]:


# Cumulative variance 
var1 = np.cumsum(np.round(var,decimals = 4)*100)
var1


# In[27]:


pca.components_


# In[29]:


# Variance plot for PCA components obtained 
plt.plot(var1,color="red")


# In[30]:


pca_values[:,0:1]


# In[31]:


# plot between PCA1 and PCA2 
x = pca_values[:,0:1]
y = pca_values[:,1:2]
#z = pca_values[:2:3]
plt.scatter(x,y)


# In[34]:


finalDf = pd.concat([pd.DataFrame(pca_values[:,0:2],columns=['pc1','pc2']), wine[['Type']]], axis = 1)


# In[35]:


finalDf


# In[43]:


import seaborn as sns
sns.scatterplot(data=finalDf,x='pc1',y='pc2',hue='Type')


# In[ ]:





# In[ ]:




