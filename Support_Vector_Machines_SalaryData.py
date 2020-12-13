#!/usr/bin/env python
# coding: utf-8

# In[1]:


# SVM Classification
import pandas as pd
import numpy as np
#from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler


# In[2]:


from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn import preprocessing


# In[3]:


from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score


# In[4]:


train_data = pd.read_csv('SalaryData_Train.csv')
test_data = pd.read_csv('SalaryData_Test.csv')


# In[5]:


label_encoder = preprocessing.LabelEncoder()
train_data['workclass']= label_encoder.fit_transform(train_data['workclass'])
train_data['education']= label_encoder.fit_transform(train_data['education'])
train_data['maritalstatus']= label_encoder.fit_transform(train_data['maritalstatus'])
train_data['occupation']= label_encoder.fit_transform(train_data['occupation'])
train_data['relationship']= label_encoder.fit_transform(train_data['relationship'])
train_data['race']= label_encoder.fit_transform(train_data['race'])
train_data['sex']= label_encoder.fit_transform(train_data['sex'])
train_data['native']= label_encoder.fit_transform(train_data['native'])
train_data['Salary']= label_encoder.fit_transform(train_data['Salary'])


# In[6]:


test_data['workclass']= label_encoder.fit_transform(test_data['workclass'])
test_data['education']= label_encoder.fit_transform(test_data['education'])
test_data['maritalstatus']= label_encoder.fit_transform(test_data['maritalstatus'])
test_data['occupation']= label_encoder.fit_transform(test_data['occupation'])
test_data['relationship']= label_encoder.fit_transform(test_data['relationship'])
test_data['race']= label_encoder.fit_transform(test_data['race'])
test_data['sex']= label_encoder.fit_transform(test_data['sex'])
test_data['native']= label_encoder.fit_transform(test_data['native'])
test_data['Salary']= label_encoder.fit_transform(test_data['Salary'])


# In[7]:


array_train = train_data.values
array_test = test_data.values

X_train = array_train[:,0:13]
Y_train = array_train[:,13]
X_test = array_test[:,0:13]
Y_test = array_test[:,13]


# ## Grid Search CV

# In[9]:


classifier = SVC()
classifier.fit(X_train, Y_train)
score = classifier.score(X_test, Y_test)


# In[10]:


score


# In[11]:


Y_pred = classifier.predict(X_test)


# In[13]:


Y_pred


# In[ ]:




