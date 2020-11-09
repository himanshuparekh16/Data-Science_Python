#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install xgboost')


# In[1]:


# First XGBoost model for Pima Indians dataset
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[4]:


# load data
dataset = loadtxt('/content/pima-indians-diabetes.data (1).csv', delimiter=",")
# split data into X and y
X = dataset[:,0:8]
Y = dataset[:,8]


# In[5]:


# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)


# In[6]:


# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)


# In[7]:


# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]


# In[8]:


predictions


# In[9]:


# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[ ]:


predictions


# ***Light GBM***

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[ ]:


# Importing the dataset
dataset = pd.read_csv('/content/pima-indians-diabetes.data.csv')
# split data into X and y
X = dataset.iloc[:,0:8]
Y = dataset.iloc[:,8]


# In[ ]:


# Splitting the dataset into the Training set and Test set

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


# In[ ]:


import lightgbm as lgb
d_train = lgb.Dataset(x_train, label=y_train)


# In[ ]:


params = {}
params['learning_rate'] = 0.003
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'binary_logloss'
params['sub_feature'] = 0.5
params['num_leaves'] = 10
params['min_data'] = 50
params['max_depth'] = 10


# In[ ]:


clf = lgb.train(params, d_train, 100)


# In[ ]:


#Prediction
y_pred=clf.predict(x_test)


# In[ ]:


predictions = [round(value) for value in y_pred]


# In[ ]:


accuracy = accuracy_score(y_test, predictions)


# In[ ]:


accuracy


# In[ ]:




