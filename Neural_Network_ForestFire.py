#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing


# In[2]:


data = pd.read_csv("forestfires.csv")


# In[3]:


data.columns


# In[4]:


label_encoder = preprocessing.LabelEncoder()
data['month']= label_encoder.fit_transform(data['month'])
data['day']= label_encoder.fit_transform(data['day'])
data['size_category']= label_encoder.fit_transform(data['size_category'])


# In[5]:


data


# In[6]:


X = data.iloc[:,0:30]
Y = data.iloc[:,30]


# In[7]:


from sklearn.model_selection import train_test_split


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X, Y)


# In[9]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)


# In[10]:


# Now apply the transformations to the data:
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[11]:


from sklearn.neural_network import MLPClassifier


# In[12]:


mlp = MLPClassifier(hidden_layer_sizes=(30,30))


# In[13]:


mlp.fit(X_train,y_train)
prediction_train=mlp.predict(X_train)
prediction_test = mlp.predict(X_test)


# In[14]:


from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,prediction_test))
np.mean(y_test==prediction_test)
np.mean(y_train==prediction_train)


# In[15]:


prediction_test


# In[ ]:




