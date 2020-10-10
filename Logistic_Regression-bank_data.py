#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.linear_model import LogisticRegression


# In[2]:


#Load the data set
bank_data = pd.read_csv("bank-full.csv")
bank_data.head()


# In[3]:


# dropping the case number columns as it is not required
bank_data.drop(['contact'],inplace=True,axis = 1)


# In[4]:


bank_data


# In[5]:


job = {'blue-collar': 1, 'management': 2, 'technician': 3, 'admin.': 4, 'services': 5, 'retired': 6, 'self-employed': 7, 'entrepreneur': 8, 'unemployed': 9, 'housemaid': 10, 'student': 11, 'unknown': 0}
marital = {'married': 1, 'single': 2, 'divorced': 3}
education = {'unknown': 0, 'primary': 1, 'secondary': 2, 'tertiary': 3}
default = {'no': 0, 'yes': 1}
housing = {'no': 0, 'yes': 1}
loan = {'no': 0, 'yes': 1}
poutcome = {'unknown': 0, 'failure': 1, 'other': 2, 'success': 3}
month = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
Target = {'no': 0, 'yes': 1}


# In[6]:


bank_data.job = [job[item] for item in bank_data.job]


# In[7]:


bank_data.marital = [marital[item] for item in bank_data.marital]


# In[8]:


bank_data.education = [education[item] for item in bank_data.education]


# In[9]:


bank_data.default = [default[item] for item in bank_data.default]


# In[10]:


bank_data.housing = [housing[item] for item in bank_data.housing]


# In[11]:


bank_data.loan = [loan[item] for item in bank_data.loan]


# In[12]:


bank_data.poutcome = [poutcome[item] for item in bank_data.poutcome]


# In[13]:


bank_data.month = [month[item] for item in bank_data.month]


# In[14]:


bank_data.Target = [Target[item] for item in bank_data.Target]


# In[15]:


bank_data


# In[16]:


bank_data.describe()


# In[17]:


bank_data.shape


# In[18]:


# Removing NA values in data set
bank_data = bank_data.dropna()
bank_data.shape


# In[19]:


# Dividing our data into input and output variables 
X = bank_data.iloc[:,0:15]
Y = bank_data.iloc[:,-1]


# In[20]:


#Logistic regression and fit the model
target = LogisticRegression()
target.fit(X,Y)


# In[21]:


#Predict for X dataset
y_pred = target.predict(X)


# In[22]:


y_pred


# In[23]:


y_pred_df= pd.DataFrame({'actual': Y, 'predicted': target.predict(X)})


# In[24]:


y_pred_df


# In[27]:


# Confusion Matrix for the model accuracy
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y,y_pred)
print (confusion_matrix)


# In[28]:


((39126+1070)/(39126+796+4219+1070))*100


# In[29]:


#Classification report
#from sklearn.metrics import classification_report
#print(classification_report(Y,y_pred))

target.predict_proba (X)[:,1]


# In[30]:


# ROC Curve

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

fpr, tpr, thresholds = roc_curve(Y, target.predict_proba (X)[:,1])

auc = roc_auc_score(Y, y_pred)

import matplotlib.pyplot as plt
plt.plot(fpr, tpr, color='red', label='logit model ( area  = %0.2f)'%auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
plt.ylabel('True Positive Rate')
plt.show()


# In[31]:


auc


# In[ ]:




