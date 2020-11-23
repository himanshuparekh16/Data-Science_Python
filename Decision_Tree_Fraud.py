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


data = pd.read_csv('Fraud_check.csv')


# In[3]:


data.head()


# In[4]:


label_encoder = preprocessing.LabelEncoder()
data['Undergrad']= label_encoder.fit_transform(data['Undergrad'])
data['Marital.Status']= label_encoder.fit_transform(data['Marital.Status'])
data['Urban']= label_encoder.fit_transform(data['Urban'])


# In[5]:


conditions = [
    (data['Taxable.Income'] <= 30000),
    (data['Taxable.Income'] > 30000)
    ]


# In[6]:


values = ['Risky', 'Safe']


# In[7]:


data['Taxable Income (Safe/Risky)'] = np.select(conditions, values)


# In[ ]:





# In[8]:


data['Taxable Income (Safe/Risky)']= label_encoder.fit_transform(data['Taxable Income (Safe/Risky)'])


# In[9]:


data


# In[10]:


x=data.iloc[:,0:6]
y=data['Taxable Income (Safe/Risky)']


# In[11]:


# Splitting data into training and testing data set
x_train, x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=40)


# ### Building Decision Tree Classifier using Entropy Criteria

# In[12]:


model = DecisionTreeClassifier(criterion = 'entropy',max_depth=3)
model.fit(x_train,y_train)


# In[13]:


#PLot the decision tree
tree.plot_tree(model);


# In[14]:


fn=['Undergrad','Marital Status','City Population','Work Exp','Urban']
cn=['Risky', 'Safe']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(model,
               feature_names = fn, 
               class_names=cn,
               filled = True);


# In[15]:


preds = model.predict(x_test) # predicting on test data set 
pd.Series(preds).value_counts() # getting the count of each category 


# In[16]:


preds


# In[17]:


pd.crosstab(y_test,preds) # getting the 2 way table to understand the correct and wrong predictions


# In[18]:


# Accuracy 
np.mean(preds==y_test)


# ### Building Decision Tree Classifier (CART) using Gini Criteria

# In[19]:


from sklearn.tree import DecisionTreeClassifier
model_gini = DecisionTreeClassifier(criterion='gini', max_depth=3)


# In[20]:


model_gini.fit(x_train, y_train)


# In[21]:


#Prediction and computing the accuracy
pred=model.predict(x_test)
np.mean(preds==y_test)


# #### Decision Tree Regression Example

# In[22]:


# Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor


# In[31]:


array = data.values
X = array[:,0:6]
y = array[:,6]


# In[32]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)


# In[33]:


model = DecisionTreeRegressor()
model.fit(X_train, y_train)


# In[34]:


#Find the accuracy
model.score(X_test,y_test)


# In[ ]:




