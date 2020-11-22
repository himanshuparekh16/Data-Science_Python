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


com_data.head()


# In[6]:


conditions = [
    (com_data['Sales'] <= 8),
    (com_data['Sales'] > 8)
    ]


# In[7]:


values = ['Low', 'High']


# In[8]:


com_data['Sales (High/Low)'] = np.select(conditions, values)


# In[9]:


com_data['Sales (High/Low)'] = label_encoder.fit_transform(com_data['Sales (High/Low)'])


# In[10]:


com_data


# In[11]:


x=com_data.iloc[:,1:11]
y=com_data['Sales (High/Low)']


# In[12]:


x_train, x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=40)


# In[13]:


model = DecisionTreeClassifier(criterion = 'entropy',max_depth=3)
model.fit(x_train,y_train)


# In[14]:


#PLot the decision tree
tree.plot_tree(model);


# In[15]:


fn=['Comp Price','Income','Advertising','Population','Price','Shelveloc','Age','Education','Urban','US']
cn=['High','Low']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(model,
               feature_names = fn, 
               class_names=cn,
               filled = True);


# In[16]:


#Predicting on test data
preds = model.predict(x_test) # predicting on test data set 
pd.Series(preds).value_counts() # getting the count of each category 


# In[17]:


preds


# In[18]:


pd.crosstab(y_test,preds)


# In[19]:


np.mean(preds==y_test)


# In[20]:


from sklearn.tree import DecisionTreeClassifier
model_gini = DecisionTreeClassifier(criterion='gini', max_depth=3)


# In[21]:


model_gini.fit(x_train, y_train)


# In[22]:


#Prediction and computing the accuracy
pred=model.predict(x_test)
np.mean(preds==y_test)


# In[23]:


# Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor


# In[37]:


array = com_data.values
X = array[:,0:11]
y = array[:,11]


# In[38]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)


# In[39]:


model = DecisionTreeRegressor()
model.fit(X_train, y_train)


# In[40]:


#Find the accuracy
model.score(X_test,y_test)


# In[ ]:




