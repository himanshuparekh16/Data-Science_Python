#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets  
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import  DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn import preprocessing


# In[2]:


data = pd.read_csv("Fraud_check.csv")
data.head()


# In[3]:


label_encoder = preprocessing.LabelEncoder()
data['Undergrad']= label_encoder.fit_transform(data['Undergrad'])
data['Marital.Status']= label_encoder.fit_transform(data['Marital.Status'])
data['Urban']= label_encoder.fit_transform(data['Urban'])


# In[4]:


conditions = [
    (data['Taxable.Income'] <= 30000),
    (data['Taxable.Income'] > 30000)
    ]


# In[5]:


values = ['Risky', 'Safe']


# In[6]:


data['Taxable Income (Safe/Risky)'] = np.select(conditions, values)


# In[7]:


data = data.drop(['Taxable.Income'], axis=1)


# In[8]:


data


# ### Data PreProcessing

# In[9]:


from sklearn.model_selection import KFold


# In[10]:


x=data.iloc[:,0:6]
y=data['Taxable Income (Safe/Risky)']


# In[13]:


kf = KFold(n_splits=2)
kf.get_n_splits(x)


# In[14]:


print(kf)


# In[15]:


KFold(n_splits=2, random_state=None, shuffle=False)


# In[17]:


# Splitting data into training and testing data set
x_train, x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=40)


# In[ ]:





# In[18]:


data.columns
colnames = list(data.columns)
predictors = colnames[:5]
target = colnames[5]


# In[19]:


X = data[predictors]
Y = data[target]


# In[20]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=3,oob_score=True,n_estimators=15,criterion="entropy")


# In[21]:


np.shape(data)


# In[22]:


rf.fit(X,Y) 
rf.estimators_ 
rf.classes_ 
rf.n_classes_ 
rf.n_features_  

rf.n_outputs_

rf.oob_score_
rf.predict(X)


# In[23]:


data['rf_pred'] = rf.predict(X)
cols = ['rf_pred','Taxable Income (Safe/Risky)']
data[cols].head()
data["Taxable Income (Safe/Risky)"]


# In[24]:


from sklearn.metrics import confusion_matrix
confusion_matrix(data['Taxable Income (Safe/Risky)'],data['rf_pred']) # Confusion matrix


# In[25]:


pd.crosstab(data['Taxable Income (Safe/Risky)'],data['rf_pred'])


# In[26]:


print("Accuracy",(114+475)/(114+475+1+10)*100)


# In[30]:


data["rf_pred"]


# In[ ]:




