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


com_data = pd.read_csv('Company_Data.csv')


# In[3]:


com_data


# In[4]:


label_encoder = preprocessing.LabelEncoder()
com_data['ShelveLoc']= label_encoder.fit_transform(com_data['ShelveLoc'])
com_data['Urban']= label_encoder.fit_transform(com_data['Urban'])
com_data['US']= label_encoder.fit_transform(com_data['US'])


# In[5]:


conditions = [
    (com_data['Sales'] <= 8),
    (com_data['Sales'] > 8)
    ]


# In[6]:


values = ['Low', 'High']


# In[7]:


com_data['Sales (High/Low)'] = np.select(conditions, values)


# In[8]:


com_data


# ### Data PreProcessing

# In[9]:


from sklearn.model_selection import KFold


# In[10]:


x=com_data.iloc[:,1:11]
y=com_data['Sales (High/Low)']


# In[11]:


kf = KFold(n_splits=2)
kf.get_n_splits(x)


# In[12]:


print(kf)


# In[13]:


KFold(n_splits=2, random_state=None, shuffle=False)


# In[14]:


x_train, x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=40)


# In[ ]:





# In[15]:


com_data.columns
colnames = list(com_data.columns)
predictors = colnames[1:11]
target = colnames[11]


# In[16]:


X = com_data[predictors]
Y = com_data[target]


# In[17]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=3,oob_score=True,n_estimators=15,criterion="entropy")


# In[18]:


np.shape(com_data)


# In[19]:


rf.fit(X,Y)  
rf.estimators_  
rf.classes_
rf.n_classes_ 
rf.n_features_

rf.n_outputs_

rf.oob_score_
rf.predict(X)


# In[20]:


com_data['rf_pred'] = rf.predict(X)
cols = ['rf_pred','Sales (High/Low)']
com_data[cols].head()
com_data["Sales (High/Low)"]


# In[21]:


from sklearn.metrics import confusion_matrix
confusion_matrix(com_data['Sales (High/Low)'],com_data['rf_pred']) # Confusion matrix


# In[22]:


pd.crosstab(com_data['Sales (High/Low)'],com_data['rf_pred'])


# In[23]:


print("Accuracy",(164+236)/(164+236+0+1)*100)


# In[24]:


com_data["rf_pred"]


# In[ ]:




