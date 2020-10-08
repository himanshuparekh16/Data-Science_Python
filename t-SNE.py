#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install bioinfokit')


# In[3]:


from pandas import read_csv
import pandas as pd
from sklearn.manifold import TSNE
from bioinfokit.visuz import cluster


# In[4]:


# load data
filename = 'TSNE_data.csv'
dataframe = pd.read_csv(filename)


# In[5]:


# Split-out validation dataset
array = dataframe.values
# separate array into input and output components
X = array[:,1:]
Y = array[:,0]


# In[6]:


#TSNE visualization
from bioinfokit.visuz import cluster

data_tsne = TSNE(n_components=2).fit_transform(X)
cluster.tsneplot(score=data_tsne)


# In[7]:


# get a list of categories
color_class = dataframe['diagnosis'].to_numpy()
cluster.tsneplot(score=data_tsne, colorlist=color_class, legendpos='upper right',legendanchor=(1.15, 1))

#Plot will be stored in the default directory


# In[8]:


data_tsne


# In[ ]:




