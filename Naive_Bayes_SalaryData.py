#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[5]:


from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing


# In[6]:


train_data = pd.read_csv('SalaryData_Train.csv')
test_data = pd.read_csv('SalaryData_Test.csv')


# In[7]:


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


# In[8]:


test_data['workclass']= label_encoder.fit_transform(test_data['workclass'])
test_data['education']= label_encoder.fit_transform(test_data['education'])
test_data['maritalstatus']= label_encoder.fit_transform(test_data['maritalstatus'])
test_data['occupation']= label_encoder.fit_transform(test_data['occupation'])
test_data['relationship']= label_encoder.fit_transform(test_data['relationship'])
test_data['race']= label_encoder.fit_transform(test_data['race'])
test_data['sex']= label_encoder.fit_transform(test_data['sex'])
test_data['native']= label_encoder.fit_transform(test_data['native'])
test_data['Salary']= label_encoder.fit_transform(test_data['Salary'])


# In[9]:


array_train = train_data.values
array_test = test_data.values

X_train = array_train[:,0:13]
Y_train = array_train[:,13]
X_test = array_test[:,0:13]
Y_test = array_test[:,13]


# In[10]:


sgnb = GaussianNB()
smnb = MultinomialNB()


# In[13]:


spred_gnb = sgnb.fit(X_train,Y_train).predict(X_test)


# In[14]:


confusion_matrix(Y_test,spred_gnb)


# In[15]:


print ("Accuracy",(10759+1209)/(10759+601+2491+1209))


# In[17]:


spred_mnb = smnb.fit(X_train,Y_train).predict(X_test)


# In[18]:


confusion_matrix(Y_test,spred_mnb)


# In[19]:


print("Accuracy",(10891+780)/(10891+780+2920+780))

