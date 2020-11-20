#!/usr/bin/env python
# coding: utf-8

# In[1]:


# KNN Classification
from pandas import read_csv
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


# In[4]:


filename = 'glass.csv'
dataframe = read_csv(filename)
array = dataframe.values
X = array[:, 0:9]
Y = array[:, 9]


# In[28]:


num_folds = 25
kfold = KFold(n_splits=25)


# In[35]:


model = KNeighborsClassifier(n_neighbors=25)
results = cross_val_score(model, X, Y, cv=kfold)


# In[36]:


print(results.mean())


# ### Grid Search for Algorithm Tuning

# In[37]:


# Grid Search for Algorithm Tuning
import numpy
from pandas import read_csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


# In[38]:


n_neighbors = numpy.array(range(1,41))
param_grid = dict(n_neighbors=n_neighbors)


# In[39]:


model = KNeighborsClassifier()
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid.fit(X, Y)


# In[40]:


print(grid.best_score_)
print(grid.best_params_)


# ### Visualizing the CV results

# In[41]:


import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
# choose k between 1 to 41
k_range = range(1, 41)
k_scores = []
# use iteration to caclulator different k in models, then return the average accuracy based on the cross validation
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, Y, cv=5)
    k_scores.append(scores.mean())
# plot to see clearly
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()


# In[ ]:




