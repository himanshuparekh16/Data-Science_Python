#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.linear_model import LogisticRegression


# In[3]:


#Load the data set
claimants = pd.read_csv("claimants.csv")
claimants.head()


# In[4]:


# dropping the case number columns as it is not required
claimants.drop(["CASENUM"],inplace=True,axis = 1)


# In[5]:


#Shape of the data set
claimants.shape


# In[6]:


# Removing NA values in data set
claimants = claimants.dropna()
claimants.shape


# In[7]:


# Dividing our data into input and output variables 
X = claimants.iloc[:,1:]
Y = claimants.iloc[:,0]


# In[8]:


#Logistic regression and fit the model
classifier = LogisticRegression()
classifier.fit(X,Y)


# In[9]:


#Predict for X dataset
y_pred = classifier.predict(X)


# In[10]:


y_pred


# In[11]:


y_pred_df= pd.DataFrame({'actual': Y,
                         'predicted_prob': classifier.predict(X)})


# In[12]:


y_pred_df


# In[13]:


# Confusion Matrix for the model accuracy
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y,y_pred)
print (confusion_matrix)


# In[14]:


((381+395)/(381+197+123+395))*100


# In[15]:


#Classification report
#from sklearn.metrics import classification_report
#print(classification_report(Y,y_pred))


# In[16]:


classifier.predict_proba (X)[:,1]


# In[17]:


# ROC Curve


# In[18]:


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

fpr, tpr, thresholds = roc_curve(Y, classifier.predict_proba (X)[:,1])

auc = roc_auc_score(Y, y_pred)

import matplotlib.pyplot as plt
plt.plot(fpr, tpr, color='red', label='logit model ( area  = %0.2f)'%auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
plt.ylabel('True Positive Rate')
plt.show()


# In[19]:


auc


# In[ ]:




