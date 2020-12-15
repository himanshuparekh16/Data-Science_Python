#!/usr/bin/env python
# coding: utf-8

# In[2]:


# SVM Classification
import pandas as pd
import numpy as np
#from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler


# In[3]:


from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn import preprocessing


# In[4]:


from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score


# In[5]:


data = pd.read_csv('forestfires.csv')


# In[7]:


label_encoder = preprocessing.LabelEncoder()
data['month']= label_encoder.fit_transform(data['month'])
data['day']= label_encoder.fit_transform(data['day'])
data['size_category']= label_encoder.fit_transform(data['size_category'])


# In[8]:


array = data.values


# In[10]:


X = array[:,0:30]
Y = array[:,30]


# In[11]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3,random_state=230)


# In[13]:


X_train.shape, Y_train.shape, X_test.shape, Y_test.shape


# ## Grid Search CV

# In[15]:


clf = SVC()
param_grid = [{'kernel':['rbf'],'gamma':[100,50,10,0.5],'C':[10,0.1,15] }]
gsv = GridSearchCV(clf,param_grid,cv=10)
gsv.fit(X_train,Y_train)


# In[16]:


gsv.best_params_ , gsv.best_score_ 


# In[17]:


clf = SVC(C= 10, gamma = 0.5)
clf.fit(X_train , Y_train)


# In[19]:


Y_pred = clf.predict(X_test)
acc = accuracy_score(Y_test, Y_pred) * 100
print("Accuracy =", acc)
confusion_matrix(Y_test, Y_pred)


# In[21]:


Y_pred


# In[ ]:




