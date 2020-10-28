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


# import some data to play with
iris = pd.read_csv('iris.csv',index_col=0)


# In[3]:


iris.head()


# In[4]:


#Complete Iris dataset
label_encoder = preprocessing.LabelEncoder()
iris['Species']= label_encoder.fit_transform(iris['Species']) 


# In[5]:


iris.head()


# In[6]:


x=iris.iloc[:,0:4]
y=iris['Species']


# In[7]:


x


# In[8]:


y


# In[9]:


iris['Species'].unique()


# In[10]:


iris.Species.value_counts()


# In[11]:


colnames = list(iris.columns)
colnames


# In[13]:


# Splitting data into training and testing data set
x_train, x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=40)


# ### Building Decision Tree Classifier using Entropy Criteria

# In[14]:


model = DecisionTreeClassifier(criterion = 'entropy',max_depth=3)
model.fit(x_train,y_train)


# In[15]:


#PLot the decision tree
tree.plot_tree(model);


# In[16]:


fn=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
cn=['setosa', 'versicolor', 'virginica']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(model,
               feature_names = fn, 
               class_names=cn,
               filled = True);


# In[19]:


#Predicting on test data
preds = model.predict(x_test) # predicting on test data set 
pd.Series(preds).value_counts() # getting the count of each category 


# In[18]:


preds


# In[20]:


pd.crosstab(y_test,preds) # getting the 2 way table to understand the correct and wrong predictions


# In[21]:


# Accuracy 
np.mean(preds==y_test)


# In[ ]:





# ### Building Decision Tree Classifier (CART) using Gini Criteria

# In[22]:


from sklearn.tree import DecisionTreeClassifier
model_gini = DecisionTreeClassifier(criterion='gini', max_depth=3)


# In[23]:


model_gini.fit(x_train, y_train)


# In[24]:


#Prediction and computing the accuracy
pred=model.predict(x_test)
np.mean(preds==y_test)


# #### Decision Tree Regression Example

# In[28]:


# Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor


# In[25]:


array = iris.values
X = array[:,0:3]
y = array[:,3]


# In[26]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)


# In[29]:


model = DecisionTreeRegressor()
model.fit(X_train, y_train)


# In[30]:


#Find the accuracy
model.score(X_test,y_test)


# In[ ]:




