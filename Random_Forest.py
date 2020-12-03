#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


# Reading the Diabetes Data #################
Diabetes = pd.read_csv("Diabetes_RF.csv")
Diabetes.head()


# In[3]:


Diabetes.columns
colnames = list(Diabetes.columns)
predictors = colnames[:8]
target = colnames[8]


# In[4]:


X = Diabetes[predictors]
Y = Diabetes[target]


# In[5]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=3,oob_score=True,n_estimators=15,criterion="entropy")


# In[6]:


# n_estimators -> Number of trees ( you can increase for better accuracy)
# n_jobs -> Parallelization of the computing and signifies the number of jobs 
# running parallel for both fit and predict
# oob_score = True means model has done out of box sampling to make predictions

np.shape(Diabetes) # 768,9 => Shape 


# In[7]:


#### Attributes that comes along with RandomForest function
rf.fit(X,Y) # Fitting RandomForestClassifier model from sklearn.ensemble 
rf.estimators_ # 
rf.classes_ # class labels (output)
rf.n_classes_ # Number of levels in class labels 
rf.n_features_  # Number of input features in model 8 here.

rf.n_outputs_ # Number of outputs when fit performed

rf.oob_score_  # 0.72916
rf.predict(X)


# In[8]:


Diabetes['rf_pred'] = rf.predict(X)
cols = ['rf_pred',' Class variable']
Diabetes[cols].head()
Diabetes[" Class variable"]


# In[9]:


from sklearn.metrics import confusion_matrix
confusion_matrix(Diabetes[' Class variable'],Diabetes['rf_pred']) # Confusion matrix


# In[10]:


pd.crosstab(Diabetes[' Class variable'],Diabetes['rf_pred'])


# In[11]:


print("Accuracy",(497+268)/(497+268+0+3)*100)


# In[12]:


# Accuracy is 99.609375
Diabetes["rf_pred"]


# In[15]:


######################### IRIS data set ##############3
iris = pd.read_csv("iris.csv")
iX=iris[["Sepal.Length","Sepal.Width","Petal.Length","Petal.Width"]]
iy=iris[["Species"]]


# In[16]:


from sklearn.ensemble import RandomForestClassifier
rfiris = RandomForestClassifier(n_jobs=4,oob_score=True,n_estimators=100,criterion="entropy")
rfiris.fit(iX,iy)
iris["rf_pred"] = rfiris.predict(iX)


# In[17]:


from sklearn.metrics import confusion_matrix
confusion_matrix(iris["Species"],iris["rf_pred"]) # 100 Percent 


# In[ ]:





# In[19]:


####################### SALARY Data #################

salary_train = pd.read_csv("SalaryData_Train.csv")
salary_test = pd.read_csv("SalaryData_Test.csv")


# In[ ]:


colnames = salary_train.columns
len(colnames[0:13])
trainX = salary_train[colnames[0:13]]
trainY = salary_train[colnames[13]]
rfsalary = RandomForestClassifier(n_jobs=2,oob_score=True,n_estimators=15,criterion="entropy")
rfsalary.fit(trainX,trainY) # Error Can not convert a string into float means we have to use LabelEncoder()


# In[ ]:


# Considering only the string data type columns and 
string_columns=["workclass","education","maritalstatus","occupation","relationship","race","sex","native"]
from sklearn import preprocessing
for i in string_columns:
    number = preprocessing.LabelEncoder()
    trainX[i] = number.fit_transform(trainX[i])


# In[ ]:


rfsalary.fit(trainX,trainY)


# In[ ]:


# Training Accuracy
trainX["rf_pred"] = rfsalary.predict(trainX)
confusion_matrix(trainY,trainX["rf_pred"]) # Confusion matrix


# In[ ]:


# Accuracy
print ("Accuracy",(22321+6954)/(22321+332+554+6954)) # 97.06


# In[ ]:


# Accuracy on testing data 
testX = salary_test[colnames[0:13]]
testY = salary_test[colnames[13]]


# In[ ]:


# Converting the string values in testing data into float
for i in string_columns:
    number = preprocessing.LabelEncoder()
    testX[i] = number.fit_transform(testX[i])
testX["rf_pred"] = rfsalary.predict(testX)
confusion_matrix(testY,testX["rf_pred"])


# In[ ]:


# Accuracy 
print ("Accuracy",(10359+2283)/(10359+1001+1417+2283)) # 83.94

