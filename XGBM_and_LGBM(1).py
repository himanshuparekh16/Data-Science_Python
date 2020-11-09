#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install xgboost')


# In[4]:


# First XGBoost model for Pima Indians dataset
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[ ]:


# load data
dataset = loadtxt('/content/pima-indians-diabetes.data (1).csv', delimiter=",")
# split data into X and y
X = dataset[:,0:8]
Y = dataset[:,8]


# In[ ]:


# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)


# In[ ]:


# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)


# In[ ]:


# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]


# In[ ]:


predictions


# In[ ]:


# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[ ]:


predictions


# ***Light GBM***

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


# Importing the dataset
dataset = pd.read_csv('/content/pima-indians-diabetes.data.csv')
# split data into X and y
X = dataset.iloc[:,0:8]
Y = dataset.iloc[:,8]


# In[5]:


# Splitting the dataset into the Training set and Test set

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


# In[6]:


import lightgbm as lgb
d_train = lgb.Dataset(x_train, label=y_train)


# In[7]:


params = {}
params['learning_rate'] = 0.003
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'binary_logloss'
params['sub_feature'] = 0.5
params['num_leaves'] = 10
params['min_data'] = 50
params['max_depth'] = 10


# In[23]:


clf = lgb.train(params, d_train, 1000)


# In[24]:


#Prediction
y_pred=clf.predict(x_test)


# In[25]:


predictions = [round(value) for value in y_pred]


# In[26]:


accuracy = accuracy_score(y_test, predictions)


# In[27]:


accuracy


# In[ ]:




