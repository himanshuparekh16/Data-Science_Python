#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# SVM Classification
import pandas as pd
import numpy as np
#from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score


# In[ ]:


filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.3,random_state=230)


# In[ ]:


X_train.shape, y_train.shape, X_test.shape, y_test.shape


# ### Grid Search CV

# In[ ]:


clf = SVC()
param_grid = [{'kernel':['rbf'],'gamma':[100,50,10,0.5],'C':[10,0.1,15] }]
gsv = GridSearchCV(clf,param_grid,cv=10)
gsv.fit(X_train,y_train)


# In[ ]:


gsv.best_params_ , gsv.best_score_ 


# In[ ]:


clf = SVC(C= 10, gamma = 50)
clf.fit(X_train , y_train)


# In[ ]:


y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred) * 100
print("Accuracy =", acc)
confusion_matrix(y_test, y_pred)


# In[ ]:


y_pred


# In[ ]:




