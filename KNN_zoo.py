#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Libraries 
import pandas as pd
import numpy as np


# In[2]:


data = pd.read_csv("zoo.csv")


# In[3]:


# Training and Test data using 
from sklearn.model_selection import train_test_split
train,test = train_test_split(data,test_size = 0.2) # 0.2 => 20 percent of entire data


# In[4]:


# KNN using sklearn 
# Importing Knn algorithm from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier as KNC


# In[5]:


# for 3 nearest neighbours 
neigh = KNC(n_neighbors= 3)


# In[6]:


# Fitting with training data 
neigh.fit(train.iloc[:,1:17],train.iloc[:,17])


# In[7]:


# train accuracy 
train_acc = np.mean(neigh.predict(train.iloc[:,1:17])==train.iloc[:,17]) # 97.5 %


# In[9]:


# test accuracy
test_acc = np.mean(neigh.predict(test.iloc[:,1:17])==test.iloc[:,17]) # 100%


# In[11]:


# for 5 nearest neighbours
neigh = KNC(n_neighbors=5)


# In[12]:


# fitting with training data
neigh.fit(train.iloc[:,1:17],train.iloc[:,17])


# In[13]:


# train accuracy 
train_acc = np.mean(neigh.predict(train.iloc[:,1:17])==train.iloc[:,17]) # 93.75%


# In[15]:


# test accuracy
test_acc = np.mean(neigh.predict(test.iloc[:,1:17])==test.iloc[:,17]) # 95.23%


# In[17]:


# creating empty list variable 
acc = []


# In[18]:


# running KNN algorithm for 3 to 50 nearest neighbours(odd numbers) and 
# storing the accuracy values 
 
for i in range(3,50,2):
    neigh = KNC(n_neighbors=i)
    neigh.fit(train.iloc[:,1:17],train.iloc[:,17])
    train_acc = np.mean(neigh.predict(train.iloc[:,1:17])==train.iloc[:,17])
    test_acc = np.mean(neigh.predict(test.iloc[:,1:17])==test.iloc[:,17])
    acc.append([train_acc,test_acc])


# In[19]:


import matplotlib.pyplot as plt # library to do visualizations


# In[20]:


# train accuracy plot 
plt.plot(np.arange(3,50,2),[i[0] for i in acc],"bo-")


# In[21]:


# test accuracy plot
plt.plot(np.arange(3,50,2),[i[1] for i in acc],"ro-")


# In[22]:


plt.legend(["train","test"])


# In[ ]:




